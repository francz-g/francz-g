"""Tira de fotogramas (para revisar transiciones). Uso: strip.py VIDEO OUT.jpg f1,f2,... [thumb_w]"""
import sys, cv2, numpy as np
v, out, fl = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(',')]
tw = int(sys.argv[4]) if len(sys.argv) > 4 else 150
cap = cv2.VideoCapture(v); ims = []
for f in fl:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = cap.read()
    if not ok: continue
    im = cv2.resize(fr, (tw, int(fr.shape[0] * tw / fr.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.putText(im, str(f), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA); ims.append(im)
cols = int(sys.argv[5]) if len(sys.argv) > 5 else len(ims)
rows = [np.hstack(ims[i:i + cols] + [np.zeros_like(ims[0])] * (cols - len(ims[i:i + cols]))) for i in range(0, len(ims), cols)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
