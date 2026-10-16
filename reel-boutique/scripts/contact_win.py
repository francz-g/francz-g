"""Hoja de contacto de una ventana: contact_win.py PROXY OUT.jpg t0 t1 step [cols] [thumb_w]"""
import cv2, sys, numpy as np
f, out, t0, t1, step = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
cols = int(sys.argv[6]) if len(sys.argv) > 6 else 8; tw = int(sys.argv[7]) if len(sys.argv) > 7 else 150
cap = cv2.VideoCapture(f); fps = cap.get(cv2.CAP_PROP_FPS); ims = []; t = t0
while t <= t1 + 1e-6:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, fr = cap.read()
    if not ok: break
    im = cv2.resize(fr, (tw, int(fr.shape[0] * tw / fr.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.rectangle(im, (0, 0), (tw, 16), (0, 0, 0), -1)
    cv2.putText(im, f"{t:.2f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    ims.append(im); t += step
h = ims[0].shape[0]; rows = (len(ims) + cols - 1) // cols
sheet = np.full((rows * h, cols * tw, 3), 25, np.uint8)
for i, im in enumerate(ims):
    r, c = divmod(i, cols); sheet[r*h:(r+1)*h, c*tw:(c+1)*tw] = im
cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 84])
