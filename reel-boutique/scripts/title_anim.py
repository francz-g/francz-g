"""Título de apertura "outfit inspo" sobre la capa PNG del Reel (escribe sobre los primeros fotogramas).

Uso: title_anim.py OVERLAY_DIR [script_word] [caps_word] [fps]
Secuencia (s):
  0,15-0,95  "outfit" en caligrafía dorada se escribe de izquierda a derecha (sincronizado con el brillo sonoro)
  0,45/0,80/1,05  pequeños destellos dorados
  0,55-1,25  "INSPO" en marfil aparece letra a letra
  2,85-3,40  el título sale hacia arriba con desenfoque suave (antes del primer barrido)
Zona segura de Instagram: el bloque queda entre el 60 % y el 77 % del alto.
"""
import os, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(__file__))
from logo import FONTS, IVORY, GOLD

W, H = 1080, 1920
CHOC = np.array([52, 33, 22], np.float32) / 255


def smooth(x):
    x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)


def text_rgba(text, font, color, track=0, stroke=0):
    w = int(font.getlength(text) + track * len(text) + font.size)
    im = Image.new('L', (w + 80, int(font.size * 1.8)), 0); d = ImageDraw.Draw(im); x = 40
    for ch in text:
        d.text((x, int(font.size * 0.3)), ch, font=font, fill=255, stroke_width=stroke, stroke_fill=255); x += font.getlength(ch) + track
    a = np.array(im.crop(im.getbbox())).astype(np.float32) / 255
    rgba = np.zeros(a.shape + (4,), np.float32); rgba[..., :3] = np.array(color, np.float32) / 255; rgba[..., 3] = a
    return rgba


def star(size):
    """Destello de 4 puntas con halo suave (RGBA, dorado claro)."""
    s = size; c = s / 2
    im = Image.new('L', (s, s), 0); d = ImageDraw.Draw(im)
    k = s * 0.06
    d.polygon([(c, 0), (c + k, c - k), (s, c), (c + k, c + k), (c, s), (c - k, c + k), (0, c), (c - k, c - k)], fill=255)
    a = np.array(im).astype(np.float32) / 255
    glow = cv2.GaussianBlur(a, (0, 0), s * 0.08) * 0.8
    a = np.clip(a + glow, 0, 1)
    rgba = np.zeros((s, s, 4), np.float32); rgba[..., :3] = np.array([255, 240, 205], np.float32) / 255; rgba[..., 3] = a
    return rgba


def comp(dst, src, x, y, alpha=1.0):
    """src RGBA (alfa recto) sobre dst premultiplicado."""
    h, w = src.shape[:2]; x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x1 <= x0 or y1 <= y0: return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:4] * alpha
    dst[y0:y1, x0:x1, :3] = s[..., :3] * a + dst[y0:y1, x0:x1, :3] * (1 - a)
    dst[y0:y1, x0:x1, 3:4] = a + dst[y0:y1, x0:x1, 3:4] * (1 - a)


def main():
    out = sys.argv[1]
    word = sys.argv[2] if len(sys.argv) > 2 else 'outfit'
    caps = sys.argv[3] if len(sys.argv) > 3 else 'INSPO'
    fps = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
    script = ImageFont.truetype(os.path.join(FONTS, 'PinyonScript-Regular.ttf'), 240)
    jost = ImageFont.truetype(os.path.join(FONTS, 'Jost[wght].ttf'), 62); jost.set_variation_by_axes([500])
    S = text_rgba(word, script, GOLD, stroke=2)   # trazo levemente más grueso: los perfiles finos sobreviven a la compresión
    letters = [text_rgba(ch, jost, IVORY) for ch in caps]
    track = 34
    widths = [l.shape[1] for l in letters]
    adv = [jost.getlength(ch) + track for ch in caps]
    cw = sum(adv) - track
    cy_script, cy_caps = 0.655 * H, 0.745 * H
    sx, sy = (W - S.shape[1]) / 2, cy_script - S.shape[0] / 2
    cx0 = (W - cw) / 2
    stars = [(0.45, sx + S.shape[1] * 0.93, sy + S.shape[0] * 0.10, 70), (0.80, cx0 + cw + 40, cy_caps - 40, 46), (1.05, sx + S.shape[1] * 0.05, sy + S.shape[0] * 0.55, 38)]
    star_img = {s: star(s) for *_, s in stars}
    n = int(3.45 * fps) + 1
    for f in range(n):
        u = f / fps
        frame = np.zeros((H, W, 4), np.float32)
        out_m = smooth((u - 2.85) / 0.55)                      # salida
        if out_m >= 1: break
        lift = -24 * out_m; vis = 1 - out_m
        # sombra suave común (legibilidad sobre la ropa/piso)
        layer = np.zeros((H, W, 4), np.float32)
        # 1) "outfit": revelado izquierda→derecha con borde suave (efecto escritura)
        p = smooth((u - 0.15) / 0.80)
        if p > 0:
            Sm = S.copy(); w = S.shape[1]
            edge = p * (w + 120) - 60
            xs = np.arange(w, dtype=np.float32)
            mask = np.clip((edge - xs) / 60.0, 0, 1)[None, :]
            Sm[..., 3] *= mask
            comp(layer, Sm, sx, sy + lift)
        # 2) INSPO letra a letra con leve subida
        x = cx0
        for i, l in enumerate(letters):
            a = smooth((u - 0.55 - 0.09 * i) / 0.38)
            if a > 0:
                comp(layer, l, x, cy_caps - l.shape[0] / 2 + (1 - a) * 14 + lift, a)
            x += adv[i]
        # sombra a partir del texto (halo chocolate muy suave)
        sh = np.clip(cv2.GaussianBlur(layer[..., 3], (0, 0), 8) * 0.85, 0, 0.75)
        frame[..., :3] = CHOC * sh[..., None]; frame[..., 3] = sh
        frame[..., :3] = layer[..., :3] + frame[..., :3] * (1 - layer[..., 3:4])
        frame[..., 3:4] = layer[..., 3:4] + frame[..., 3:4] * (1 - layer[..., 3:4])
        # 3) destellos
        for t0, x0, y0, s in stars:
            q = (u - t0) / 0.55
            if 0 < q < 1:
                g = np.sin(np.pi * q) ** 1.5
                k = 0.4 + 0.8 * g
                img = cv2.resize(star_img[s], (max(2, int(s * k)), max(2, int(s * k))), interpolation=cv2.INTER_AREA)
                M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 25 * q, 1.0)
                img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                comp(frame, img, x0 - img.shape[1] / 2, y0 - img.shape[0] / 2 + lift, g)
        if out_m > 0:
            frame = cv2.GaussianBlur(frame, (0, 0), 0.1 + 6 * out_m)
        frame *= vis
        # despremultiplicar y fusionar con lo que ya haya en la capa (normalmente transparente)
        path = os.path.join(out, f'logo_{f:04d}.png')
        if os.path.exists(path) and not os.path.islink(path):
            prev = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255
            pa = prev[..., 3:4]
            frame[..., :3] = frame[..., :3] + prev[..., :3] * pa * (1 - frame[..., 3:4])
            frame[..., 3:4] = frame[..., 3:4] + pa * (1 - frame[..., 3:4])
        a = frame[..., 3:4]
        rgb = np.where(a > 1e-4, frame[..., :3] / np.maximum(a, 1e-4), 0)
        img = np.clip(np.concatenate([rgb, a], 2) * 255 + 0.5, 0, 255).astype(np.uint8)
        if os.path.islink(path): os.remove(path)
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    print('título en fotogramas 0 -', f)


if __name__ == '__main__':
    main()
