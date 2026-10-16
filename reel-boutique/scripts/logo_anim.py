"""Animación de cierre con el logo S&P BOUTIQUE como capa RGBA (secuencia PNG del largo total del video).

Uso: logo_anim.py OUT_DIR total_frames start_s [fps]
Secuencia (segundos desde start_s):
  0,00-0,60  degradado sutil arriba para que el logo se lea sobre la pared clara
  0,15-0,90  S y P aparecen con enfoque suave y leve subida
  0,42-1,07  el "&" dorado aparece con un ligero escalado
  0,85-1,35  la línea dorada se dibuja desde el centro
  1,05-1,95  BOUTIQUE aparece letra a letra
  1,70-2,30  destello dorado que recorre el "&" y la línea
  2,45-3,35  la imagen funde a chocolate y el logo baja al centro y crece (cierre de marca)
"""
import os, sys, math
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
import logo

W, H = 1080, 1920
CHOC = np.array([52, 33, 22], np.float32)
TOP = dict(cx=0.5, y=0.045, w=0.25)          # posición arriba (como en la captura de referencia)
CARD = dict(cx=0.5, cy=0.47, w=0.40)         # cierre centrado sobre chocolate


def smooth(x):
    x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)


def fade(u, t0, d):
    return smooth((u - t0) / d)


def layer_arrays(cap=300):
    L = logo.build(cap=cap)
    items = {k: v for k, v in L.items() if not k.startswith('_')}
    xs = [x for im, x, y in items.values()] + [x + im.width for im, x, y in items.values()]
    ys = [y for im, x, y in items.values()] + [y + im.height for im, x, y in items.values()]
    pad = int(cap * 0.25)
    ox, oy = -min(xs) + pad, -min(ys) + pad
    cw, ch = max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad
    arr = {k: (np.array(im).astype(np.float32) / 255.0, x + ox, y + oy) for k, (im, x, y) in items.items()}
    sp_w = (items['P'][1] + items['P'][0].width) - items['S'][1]       # ancho de "S&P"
    # columnas de cada letra de BOUTIQUE (para la aparición escalonada)
    b = arr['boutique'][0][..., 3]
    cols = b.max(0) > 0.05
    spans, inside = [], False
    for i, c in enumerate(cols):
        if c and not inside: start = i; inside = True
        if not c and inside: spans.append((start, i)); inside = False
    if inside: spans.append((start, len(cols)))
    return arr, (cw, ch), sp_w, spans, cap


def paste(canvas, rgba, x, y, alpha=1.0, blur=0.0, scale=1.0):
    """Pega una capa RGBA premultiplicada sobre el lienzo (también premultiplicado)."""
    im = rgba.copy()
    if scale != 1.0:
        h, w = im.shape[:2]
        im = cv2.resize(im, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        x += (w - im.shape[1]) / 2; y += (h - im.shape[0]) / 2
    pm = im.copy(); pm[..., :3] *= pm[..., 3:4]; pm *= alpha
    if blur > 0.3:
        pm = cv2.GaussianBlur(pm, (0, 0), blur)
    x, y = int(round(x)), int(round(y))
    h, w = pm.shape[:2]
    x0, y0 = max(0, x), max(0, y); x1, y1 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x1 <= x0 or y1 <= y0: return
    src = pm[y0 - y:y1 - y, x0 - x:x1 - x]
    dst = canvas[y0:y1, x0:x1]
    dst[...] = src + dst * (1 - src[..., 3:4])


def render_block(u, arr, size, spans, cap):
    """Logo completo en su lienzo maestro (premultiplicado) para el instante u."""
    cw, ch = size
    cv = np.zeros((ch, cw, 4), np.float32)
    for key, t0 in (('S', 0.15), ('P', 0.27)):
        a = fade(u, t0, 0.62)
        if a > 0:
            im, x, y = arr[key]
            paste(cv, im, x, y + (1 - a) * 0.09 * cap, a, blur=(1 - a) * 0.05 * cap)
    a = fade(u, 0.42, 0.65)
    if a > 0:
        im, x, y = arr['amp']
        amp = im.copy()
        # destello dorado diagonal
        s = (u - 1.70) / 0.6
        if 0 < s < 1:
            hh, ww = amp.shape[:2]
            gx, gy = np.meshgrid(np.arange(ww), np.arange(hh))
            c = -0.4 * ww + s * 1.8 * ww
            band = np.exp(-((gx - c + 0.45 * gy) ** 2) / (2 * (0.10 * ww) ** 2))[..., None] * 0.55
            amp[..., :3] = amp[..., :3] * (1 - band) + np.array([1.0, 0.97, 0.90]) * band
        paste(cv, amp, x, y, a, blur=(1 - a) * 0.05 * cap, scale=0.93 + 0.07 * a)
    p = fade(u, 0.85, 0.5)
    if p > 0:
        im, x, y = arr['line']
        w = im.shape[1]; vis = max(1, int(w * p)); off = (w - vis) // 2
        line = im[:, off:off + vis].copy()
        s = (u - 1.85) / 0.5
        if 0 < s < 1:
            gx = np.arange(line.shape[1])[None, :, None]
            band = np.exp(-((gx - s * line.shape[1]) ** 2) / (2 * (0.12 * w) ** 2)) * 0.6
            line[..., :3] = line[..., :3] * (1 - band) + np.array([1.0, 0.97, 0.90]) * band
        paste(cv, line, x + off, y, 1.0)
    im, x, y = arr['boutique']
    if len(spans) == 8:
        for i, (c0, c1) in enumerate(spans):
            a = fade(u, 1.05 + 0.065 * i, 0.45)
            if a > 0:
                paste(cv, im[:, c0:c1], x + c0, y + (1 - a) * 0.05 * cap, a)
    else:
        a = fade(u, 1.05, 0.8)
        if a > 0: paste(cv, im, x, y, a)
    return cv


def main():
    out, total, start = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    fps = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
    os.makedirs(out, exist_ok=True)
    arr, size, sp_w, spans, cap = layer_arrays()
    cw, ch = size
    blank = os.path.join(out, 'blank.png')
    cv2.imwrite(blank, np.zeros((H, W, 4), np.uint8))
    yy = np.arange(H, dtype=np.float32)[:, None, None] / H
    scrim_shape = np.clip(1 - yy / 0.34, 0, 1) ** 1.5  # degradado superior
    f0 = int(round(start * fps))
    for f in range(total):
        path = os.path.join(out, f'logo_{f:04d}.png')
        if f < f0:
            if os.path.lexists(path): os.remove(path)
            os.symlink('blank.png', path); continue
        u = (f - f0) / fps
        frame = np.zeros((H, W, 4), np.float32)
        m = fade(u, 2.62, 0.78)                                  # el logo baja al centro y crece
        mp = fade(u, 2.40, 0.65)                                 # la imagen funde a chocolate (antes de que el logo cruce la cara)
        scrim_a = 0.55 * fade(u, 0.0, 0.6) * (1 - mp)
        if scrim_a > 0:
            frame[..., :3] = CHOC / 255 * (scrim_a * scrim_shape)
            frame[..., 3:4] = scrim_a * scrim_shape
        if mp > 0:                                               # placa chocolate a pantalla completa
            plate = np.zeros_like(frame); plate[..., :3] = CHOC / 255 * mp; plate[..., 3] = mp
            frame = plate + frame * (1 - plate[..., 3:4])
        block = render_block(u, arr, size, spans, cap)
        # escala y posición: de arriba (TOP) al centro (CARD)
        s_top = TOP['w'] * W / sp_w; s_card = CARD['w'] * W / sp_w
        s = s_top + (s_card - s_top) * m
        bw, bh = int(cw * s), int(ch * s)
        blk = cv2.resize(block, (bw, bh), interpolation=cv2.INTER_AREA)
        # sombra suave para leerse sobre la pared clara (desaparece en el cierre sobre chocolate)
        sh_a = 0.45 * (1 - mp)
        if sh_a > 0.01:
            sh = cv2.GaussianBlur(blk[..., 3], (0, 0), max(1.0, 0.012 * bw))
            shadow = np.zeros_like(blk); shadow[..., :3] = (CHOC / 255 * 0.6) * (sh * sh_a)[..., None]; shadow[..., 3] = sh * sh_a
            blk = blk + shadow * (1 - blk[..., 3:4])
        pad = cap * 0.25 * s
        top_y = TOP['y'] * H - pad
        card_y = CARD['cy'] * H - bh / 2
        y = top_y + (card_y - top_y) * m
        x = TOP['cx'] * W - bw / 2
        xi, yi = int(round(x)), int(round(y))
        x0, y0 = max(0, xi), max(0, yi); x1, y1 = min(W, xi + bw), min(H, yi + bh)
        src = blk[y0 - yi:y1 - yi, x0 - xi:x1 - xi]
        dst = frame[y0:y1, x0:x1]
        dst[...] = src + dst * (1 - src[..., 3:4])
        # despremultiplica para PNG con alfa recto
        a = frame[..., 3:4]
        rgb = np.where(a > 1e-4, frame[..., :3] / np.maximum(a, 1e-4), 0)
        outimg = np.concatenate([rgb, a], 2)
        outimg = np.clip(outimg * 255 + 0.5, 0, 255).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(outimg, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    print('frames', total, 'logo desde', f0)


if __name__ == '__main__':
    main()
