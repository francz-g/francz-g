"""Cámara virtual: estabilización + reencuadre 9:16 siguiendo a la modelo.

Todo el análisis se hace sobre el proxy (540 px de ancho); las transformaciones se
escalan a la resolución de trabajo al renderizar.
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from scipy.ndimage import gaussian_filter1d

MD = os.path.join(os.environ.get('REEL_DIR', '.'), 'models') + '/'


def read_proxy_frames(path, f0, f1):
    """Fotogramas [f0, f1] (inclusive) del proxy en BGR."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f0))
    frames = []
    for _ in range(max(0, f0), f1 + 1):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def track_subject(frames, fps):
    """Caja de la persona por fotograma (x0, y0, x1, y1, cx, head_y) normalizada, NaN si no hay pose."""
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MD + 'pose_landmarker_full.task'),
        running_mode=vision.RunningMode.VIDEO, min_pose_detection_confidence=0.4, min_tracking_confidence=0.4))
    out = np.full((len(frames), 6), np.nan)
    for i, fr in enumerate(frames):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        r = pose.detect_for_video(img, int(i * 1000 / fps) + 1)
        if not r.pose_landmarks:
            continue
        lm = r.pose_landmarks[0]
        xs = np.array([p.x for p in lm]); ys = np.array([p.y for p in lm]); vis = np.array([p.visibility for p in lm])
        good = vis > 0.5
        if good.sum() < 6:
            continue
        head = ys[[0, 2, 5, 7, 8]].min() - 0.55 * abs((ys[11] + ys[12]) / 2 - ys[0])
        cx = np.mean(xs[[11, 12, 23, 24]])
        out[i] = [xs[good].min(), min(ys[good].min(), head), xs[good].max(), ys[good].max(), cx, head]
    pose.close()
    return out


def camera_motion(frames, boxes):
    """Movimiento global de cámara entre fotogramas consecutivos (dx, dy, da, ds) ignorando a la persona."""
    n = len(frames)
    mot = np.zeros((n, 4)); mot[:, 3] = 1.0
    H, W = frames[0].shape[:2]
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, n):
        cur = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        mask = np.full(prev.shape, 255, np.uint8)
        for b in (boxes[i - 1], boxes[i]):
            if not np.isnan(b[0]):
                x0, y0, x1, y1 = (b[:4] * [W, H, W, H]).astype(int)
                mask[max(0, y0 - 25):min(H, y1 + 25), max(0, x0 - 25):min(W, x1 + 25)] = 0
        p0 = cv2.goodFeaturesToTrack(prev, 400, 0.01, 10, mask=mask)
        if p0 is None or len(p0) < 15:
            p0 = cv2.goodFeaturesToTrack(prev, 400, 0.01, 10)
        if p0 is not None and len(p0) >= 10:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
            g0, g1 = p0[st == 1], p1[st == 1]
            if len(g0) >= 8:
                M, inl = cv2.estimateAffinePartial2D(g0, g1, method=cv2.RANSAC, ransacReprojThreshold=1.5)
                # solo se acepta si la mayoría de puntos coincide (fondo rígido); si no, se asume cámara quieta
                if M is not None and inl is not None and inl.sum() >= 20 and inl.mean() >= 0.5:
                    mot[i] = [M[0, 2], M[1, 2], np.arctan2(M[1, 0], M[0, 0]), np.hypot(M[0, 0], M[1, 0])]
        prev = cur
    return mot


def _interp_nan(a):
    a = a.copy()
    idx = np.arange(len(a)); ok = ~np.isnan(a)
    if ok.sum() == 0:
        return None
    a[~ok] = np.interp(idx[~ok], idx[ok], a[ok])
    return a


def _rot_scale(a, s, cx, cy):
    """Matriz 3x3 de rotación a y escala s alrededor de (cx, cy)."""
    c, si = np.cos(a) * s, np.sin(a) * s
    return np.array([[c, -si, cx - c * cx + si * cy], [si, c, cy - si * cx - c * cy], [0, 0, 1]])


def plan_camera(frames, fps, cam, out_aspect=9 / 16):
    """Calcula, por fotograma del proxy, la matriz 3x3 que lleva el proxy al recorte final normalizado [0,1]^2.

    cam: dict con
      stab   : sigma de suavizado en s (0 = sin estabilizar, 'lock' = trípode)
      follow : 0..1 cuánto sigue el recorte a la modelo (0 = encuadre fijo)
      zoom   : zoom base sobre el mínimo necesario (1.0 = mínimo)
      push   : zoom adicional lineal a lo largo del plano (0.05 = +5 %)
      head   : posición objetivo de la cabeza (fracción del alto del recorte) o None = centrar cuerpo
      xbias  : desplazamiento horizontal de la modelo (-0.5..0.5 del ancho del recorte)
      ycenter: centro vertical fijo (fracción del alto del proxy) si head es None
    """
    n = len(frames)
    H, W = frames[0].shape[:2]
    boxes = track_subject(frames, fps)
    mot = camera_motion(frames, boxes)
    # trayectoria acumulada
    traj = np.cumsum(np.column_stack([mot[:, 0], mot[:, 1], mot[:, 2], np.log(mot[:, 3])]), axis=0)
    stab = cam.get('stab', 0.6)
    if stab == 'lock':
        smooth = np.repeat(traj[n // 2:n // 2 + 1], n, axis=0)
    elif stab and stab > 0:
        smooth = np.column_stack([gaussian_filter1d(traj[:, k], stab * fps, mode='nearest') for k in range(4)])
    else:
        smooth = traj.copy()
    corr = smooth - traj  # corrección a aplicar a cada fotograma
    C = [np.array([[1, 0, corr[i, 0]], [0, 1, corr[i, 1]], [0, 0, 1]]) @ _rot_scale(corr[i, 2], np.exp(corr[i, 3]), W / 2, H / 2)
         for i in range(n)]
    # sujeto en coordenadas corregidas
    cx = _interp_nan(boxes[:, 4]); head = _interp_nan(boxes[:, 5]); y1 = _interp_nan(boxes[:, 3])
    if cx is None:
        cx = np.full(n, 0.5); head = np.full(n, 0.12); y1 = np.full(n, 0.95)
    subj = np.array([C[i] @ [cx[i] * W, head[i] * H, 1] for i in range(n)])[:, :2]
    feet = np.array([C[i] @ [cx[i] * W, y1[i] * H, 1] for i in range(n)])[:, 1]
    # tamaño del recorte: alto ch (px del proxy); ancho = ch * aspect
    ch_full = min(H, W / out_aspect)
    zoom = cam.get('zoom', 1.0); push = cam.get('push', 0.0)
    ramp = np.linspace(0, 1, n)
    follow = cam.get('follow', 0.6)
    fx = gaussian_filter1d(subj[:, 0], 1.2 * fps, mode='nearest')
    target_x = fx * follow + np.median(subj[:, 0]) * (1 - follow) - cam.get('xbias', 0.0) * ch_full * out_aspect
    hd = gaussian_filter1d(subj[:, 1], 1.2 * fps, mode='nearest')
    ft = gaussian_filter1d(feet, 1.2 * fps, mode='nearest')
    hd = hd * follow + np.median(subj[:, 1]) * (1 - follow)
    ft = ft * follow + np.median(feet) * (1 - follow)
    # zoom constante por plano (más el "push" opcional) para que el recorte nunca muestre bordes ni "respire"
    Cinv = [np.linalg.inv(c) for c in C]

    def overflow(i, ux_i, uy_i, cw_i, ch_i):
        corners = np.array([[ux_i + sx * cw_i / 2, uy_i + sy * ch_i / 2, 1] for sx in (-1, 1) for sy in (-1, 1)])
        src = (Cinv[i] @ corners.T).T[:, :2]
        return src

    z0 = zoom
    for _ in range(10):
        z = z0 * (1 + push * ramp)
        ch = ch_full / z; cw = ch * out_aspect
        if cam.get('head') is not None:
            cy = hd - ch * cam['head'] + ch / 2
        else:
            cy = np.full(n, cam['ycenter'] * H) if cam.get('ycenter') else (hd + ft) / 2
        ux = target_x.copy(); uy = cy.copy()
        for i in range(n):  # desplaza el centro hasta que las esquinas caigan dentro de la imagen original
            for _k in range(5):
                src = overflow(i, ux[i], uy[i], cw[i], ch[i])
                ox = max(0, -src[:, 0].min()) - max(0, src[:, 0].max() - W)
                oy = max(0, -src[:, 1].min()) - max(0, src[:, 1].max() - H)
                if abs(ox) < 0.25 and abs(oy) < 0.25:
                    break
                ux[i] += ox; uy[i] += oy
        ux = gaussian_filter1d(ux, 0.25 * fps, mode='nearest'); uy = gaussian_filter1d(uy, 0.25 * fps, mode='nearest')
        need = np.ones(n)
        for i in range(n):
            src = overflow(i, ux[i], uy[i], cw[i], ch[i])
            over = max(-src[:, 0].min() / (cw[i] / 2), (src[:, 0].max() - W) / (cw[i] / 2),
                       -src[:, 1].min() / (ch[i] / 2), (src[:, 1].max() - H) / (ch[i] / 2), 0)
            need[i] = 1 + over
        if need.max() <= 1.0005:
            break
        z0 *= need.max() * 1.004
    mats = []
    for i in range(n):
        # recorte -> [0,1]^2
        K = np.array([[1 / cw[i], 0, -(ux[i] - cw[i] / 2) / cw[i]], [0, 1 / ch[i], -(uy[i] - ch[i] / 2) / ch[i]], [0, 0, 1]])
        mats.append(K @ C[i])
    info = dict(zoom_min=float(z.min()), zoom_max=float(z.max()), shake_px=float(np.std(corr[:, 0]) + np.std(corr[:, 1])),
                pose_frac=float(np.mean(~np.isnan(boxes[:, 0]))))
    return np.array(mats), boxes, info
