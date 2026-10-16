"""Temblor de cámara medido solo en el fondo (bordes del cuadro), por bandas: vibración (<0,2 s) y bamboleo (0,2-1 s).
Uso: check_stab2.py EDL.json SHOTS_DIR [i,j,...]"""
import json, sys, os, cv2, numpy as np
from scipy.ndimage import uniform_filter1d
sys.path.insert(0, os.path.dirname(__file__))
from render_shots import handles, frame_plan, src_times
edl = json.load(open(sys.argv[1])); sdir = sys.argv[2]; fps = edl['fps']
only = {int(x) for x in sys.argv[3].split(',')} if len(sys.argv) > 3 else None
_, ns = frame_plan(edl); hs = handles(edl)

def traj(frames):
    h, w = frames[0].shape[:2]
    mask = np.zeros((h, w), np.uint8); mask[:, :int(0.22 * w)] = 255; mask[:, int(0.78 * w):] = 255; mask[:int(0.14 * h)] = 255
    tr = [np.zeros(2)]; prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for f in frames[1:]:
        cur = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); d = np.zeros(2)
        p0 = cv2.goodFeaturesToTrack(prev, 300, 0.01, 8, mask=mask)
        if p0 is not None and len(p0) > 12:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
            M, inl = cv2.estimateAffinePartial2D(p0[st == 1], p1[st == 1], method=cv2.RANSAC, ransacReprojThreshold=1.0)
            if M is not None: d = M[:, 2]
        tr.append(tr[-1] + d); prev = cur
    return np.array(tr) / w * 1080  # en píxeles de la salida (1080 de ancho)

def bands(tr):
    hf = tr - uniform_filter1d(tr, 5, axis=0)
    mid = uniform_filter1d(tr, 5, axis=0) - uniform_filter1d(tr, 30, axis=0)
    return float(np.mean(np.hypot(*hf.T))), float(np.mean(np.hypot(*mid.T)))

R = []
for i, s in enumerate(edl['shots']):
    if only is not None and i not in only: continue
    pre, post = hs[i]; ts, _ = src_times(s, fps, pre, post, ns[i])
    cap = cv2.VideoCapture(os.path.join(edl['proxy_dir'], s['clip'] + '.mp4')); sf = cap.get(cv2.CAP_PROP_FPS); b = []
    for t in ts[pre:pre + ns[i]]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * sf))); ok, fr = cap.read()
        if ok: b.append(fr)
    cap = cv2.VideoCapture(os.path.join(sdir, f'shot_{i:02d}.mkv')); a = []; k = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if pre <= k < pre + ns[i]: a.append(cv2.resize(fr, (540, 960), interpolation=cv2.INTER_AREA))
        k += 1
    (bh, bm), (ah, am) = bands(traj(b)), bands(traj(a))
    R.append((bh, bm, ah, am))
    print(f"plano {i:02d} {s['clip']}  vibración {bh:.2f}->{ah:.2f} px   bamboleo {bm:.2f}->{am:.2f} px")
R = np.array(R)
print(f"MEDIA vibración {R[:,0].mean():.2f}->{R[:,2].mean():.2f} px   bamboleo {R[:,1].mean():.2f}->{R[:,3].mean():.2f} px")
