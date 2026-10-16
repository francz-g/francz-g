"""Hojas de contacto: un fotograma cada `step` segundos, con nombre y tiempo.
Uso: contact.py PROXY_DIR OUT_DIR step [cols rows thumb_w] [clip ...]"""
import cv2, sys, os, glob, numpy as np
pdir, odir, step = sys.argv[1], sys.argv[2], float(sys.argv[3])
cols = int(sys.argv[4]) if len(sys.argv) > 4 else 8
rows = int(sys.argv[5]) if len(sys.argv) > 5 else 3
tw = int(sys.argv[6]) if len(sys.argv) > 6 else 180
only = set(sys.argv[7:])
os.makedirs(odir, exist_ok=True)
for f in sorted(glob.glob(os.path.join(pdir, '*.mp4'))):
    name = os.path.basename(f)[:-4]
    if only and name not in only: continue
    cap = cv2.VideoCapture(f); fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n / fps; thumbs = []; t = 0.0
    while t < dur - 0.05:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, fr = cap.read()
        if not ok: break
        th = int(round(fr.shape[0] * tw / fr.shape[1]))
        im = cv2.resize(fr, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(im, (0, 0), (tw, 18), (0, 0, 0), -1)
        cv2.putText(im, f"{name[4:]} {t:5.1f}s", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(im); t += step
    per = cols * rows
    for p in range(0, len(thumbs), per):
        chunk = thumbs[p:p + per]
        h = chunk[0].shape[0]
        sheet = np.full((rows * h, cols * tw, 3), 30, np.uint8)
        for i, im in enumerate(chunk):
            r, c = divmod(i, cols); sheet[r*h:r*h+im.shape[0], c*tw:c*tw+tw] = im
        used_rows = (len(chunk) + cols - 1) // cols
        cv2.imwrite(os.path.join(odir, f"{name}_{p//per:02d}.jpg"), sheet[:used_rows*h], [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(name, f"{dur:.1f}s", len(thumbs), 'thumbs')
