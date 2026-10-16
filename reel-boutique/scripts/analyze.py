"""Análisis por clip sobre los proxies: pose (encuadre, giros), sonrisa, movimiento de cámara, nitidez, luz.
Uso: analyze.py PROXY.mp4 OUT.npz [step]"""
import sys, os, cv2, numpy as np, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
MD = os.path.join(os.environ.get('REEL_DIR', '.'), 'models') + '/'
f, out = sys.argv[1], sys.argv[2]
step = int(sys.argv[3]) if len(sys.argv) > 3 else 3
cap = cv2.VideoCapture(f); fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MD + 'pose_landmarker_full.task'), running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.4, min_tracking_confidence=0.4))
face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MD + 'face_landmarker.task'), running_mode=vision.RunningMode.VIDEO,
    output_face_blendshapes=True, min_face_detection_confidence=0.4))
keys = ['t', 'has_pose', 'x0', 'y0', 'x1', 'y1', 'cx', 'facing', 'feet', 'head_y', 'has_face', 'smile', 'yaw', 'eyes_open',
        'cam_dx', 'cam_dy', 'cam_da', 'cam_ds', 'sharp', 'luma', 'r', 'g', 'b', 'subj_motion']
rec = {k: [] for k in keys}
prev_gray = None; prev_box = None; prev_lm = None
i = 0
while True:
    ok, fr = cap.read()
    if not ok: break
    if i % step: i += 1; continue
    t = i / fps; ts = int(t * 1000)
    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB); H, W = fr.shape[:2]
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    pr = pose.detect_for_video(img, ts)
    row = dict.fromkeys(keys, np.nan); row['t'] = t; row['has_pose'] = 0; row['has_face'] = 0
    box = None
    if pr.pose_landmarks:
        lm = pr.pose_landmarks[0]
        xs = np.array([p.x for p in lm]); ys = np.array([p.y for p in lm]); vis = np.array([p.visibility for p in lm])
        good = vis > 0.5
        if good.sum() >= 6:
            row['has_pose'] = 1
            sh_len = abs(ys[11] + ys[12] - ys[23] - ys[24]) / 2 + 1e-6
            head_top = ys[[0, 2, 5, 7, 8]].min() - 0.55 * abs((ys[11] + ys[12]) / 2 - ys[0])
            x0, x1 = xs[good].min(), xs[good].max(); y0 = min(ys[good].min(), head_top); y1 = ys[good].max()
            row.update(x0=x0, y0=y0, x1=x1, y1=y1, cx=(xs[11] + xs[12] + xs[23] + xs[24]) / 4, head_y=head_top,
                       facing=(xs[11] - xs[12]) / sh_len, feet=float(np.mean(vis[[27, 28, 29, 30, 31, 32]] > 0.5) * (ys[[27, 28]].max() < 0.985)))
            box = (int(max(0, x0 * W - 20)), int(max(0, y0 * H - 20)), int(min(W, x1 * W + 20)), int(min(H, y1 * H + 20)))
            pts = np.stack([xs, ys], 1)[[11, 12, 13, 14, 15, 16, 23, 24, 25, 26]]
            if prev_lm is not None: row['subj_motion'] = float(np.mean(np.linalg.norm(pts - prev_lm, axis=1)))
            prev_lm = pts
    fres = face.detect_for_video(img, ts)
    if fres.face_landmarks:
        row['has_face'] = 1
        bs = {c.category_name: c.score for c in fres.face_blendshapes[0]}
        row['smile'] = (bs.get('mouthSmileLeft', 0) + bs.get('mouthSmileRight', 0)) / 2
        row['eyes_open'] = 1 - (bs.get('eyeBlinkLeft', 0) + bs.get('eyeBlinkRight', 0)) / 2
        fl = fres.face_landmarks[0]; fx = np.array([p.x for p in fl])
        row['yaw'] = (fl[1].x - (fx.min() + fx.max()) / 2) / (fx.max() - fx.min() + 1e-6)
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        mask = np.full(gray.shape, 255, np.uint8)
        for b in (box, prev_box):
            if b: mask[b[1]:b[3], b[0]:b[2]] = 0
        p0 = cv2.goodFeaturesToTrack(prev_gray, 300, 0.01, 12, mask=mask)
        if p0 is not None and len(p0) >= 12:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=3)
            g0, g1 = p0[st == 1], p1[st == 1]
            if len(g0) >= 10:
                M, inl = cv2.estimateAffinePartial2D(g0, g1, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                if M is not None:
                    row['cam_dx'] = M[0, 2] / W; row['cam_dy'] = M[1, 2] / W
                    row['cam_da'] = float(np.arctan2(M[1, 0], M[0, 0])); row['cam_ds'] = float(np.hypot(M[0, 0], M[1, 0]))
    roi = gray[box[1]:box[3], box[0]:box[2]] if box else gray
    row['sharp'] = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else np.nan
    row['luma'] = float(gray.mean()) / 255
    m = fr.reshape(-1, 3).mean(0); row['b'], row['g'], row['r'] = m / 255
    for k in keys: rec[k].append(row[k])
    prev_gray, prev_box = gray, box; i += 1
pose.close(); face.close()
np.savez(out, fps=fps, n=n, W=W, H=H, step=step, **{k: np.array(v, float) for k, v in rec.items()})
print(os.path.basename(f), 'frames', n, 'samples', len(rec['t']))
