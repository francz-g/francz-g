"""Compara el temblor de cámara (alta frecuencia del movimiento del fondo) antes/después por plano.
Uso: check_stab.py EDL.json SHOTS_DIR"""
import json, sys, os, cv2, numpy as np
from scipy.ndimage import uniform_filter1d
sys.path.insert(0, os.path.dirname(__file__))
from render_shots import handles, frame_plan, src_times
edl = json.load(open(sys.argv[1])); sdir = sys.argv[2]; fps = edl['fps']
_, ns = frame_plan(edl); hs = handles(edl)

def jitter(frames):
    tr = [np.zeros(2)]; prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for f in frames[1:]:
        cur = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev, 300, 0.01, 10)
        d = np.zeros(2)
        if p0 is not None and len(p0) > 10:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None)
            M, inl = cv2.estimateAffinePartial2D(p0[st == 1], p1[st == 1], method=cv2.RANSAC)
            if M is not None: d = M[:, 2]
        tr.append(tr[-1] + d); prev = cur
    tr = np.array(tr); k = max(3, int(fps * 0.5))
    return float(np.mean(np.hypot(*(tr - uniform_filter1d(tr, k, axis=0)).T)) / frames[0].shape[1] * 100)

tot_b, tot_a = [], []
for i, s in enumerate(edl['shots']):
    pre, post = hs[i]
    ts, _ = src_times(s, fps, pre, post, ns[i])
    cap = cv2.VideoCapture(os.path.join(edl['proxy_dir'], s['clip'] + '.mp4')); sf = cap.get(cv2.CAP_PROP_FPS)
    before = []
    for t in ts[pre:pre + ns[i]]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * sf))); ok, fr = cap.read()
        if ok: before.append(cv2.resize(fr, (270, 480)))
    cap = cv2.VideoCapture(os.path.join(sdir, f'shot_{i:02d}.mkv')); after = []
    k = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if pre <= k < pre + ns[i]: after.append(cv2.resize(fr, (270, 480)))
        k += 1
    b, a = jitter(before), jitter(after)
    tot_b.append(b); tot_a.append(a)
    print(f"plano {i:02d} {s['clip']}  temblor antes {b:.2f}%  después {a:.2f}%")
print(f"MEDIA antes {np.mean(tot_b):.2f}%  después {np.mean(tot_a):.2f}%  (reducción {100 * (1 - np.mean(tot_a) / np.mean(tot_b)):.0f} %)")
