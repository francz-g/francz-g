"""Recreación tipográfica del logo S&P BOUTIQUE (blanco marfil + dorado champán) en capas RGBA."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = os.path.join(os.path.dirname(__file__), '..', 'fonts')
IVORY = (252, 251, 247)
GOLD = (221, 196, 145)


def _font(name, size, wght, opsz=24):
    f = ImageFont.truetype(os.path.join(FONTS, name), size)
    axes = [a['name'] for a in f.get_variation_axes()] if hasattr(f, 'get_variation_axes') else []
    vals = [wght if n in (b'Weight', 'Weight') else opsz for n in axes]
    if vals:
        f.set_variation_by_axes(vals)
    return f


def _ampersand(font):
    """'&' de Bodoni Moda con la bandera del brazo derecho sustituida por una bola, como en el logo original."""
    A, top = _glyph('&', font, GOLD)
    a = np.array(A)[..., 3].astype(np.float32); h, w = a.shape
    flag = [y for y in range(int(h * 0.3), int(h * 0.55)) if (a[y, int(w * 0.68):] > 128).sum() > w * 0.15]
    if flag:
        y0, y1 = min(flag) - 2, max(flag) + 2
        yb = min(h - 1, y1 + 4)
        ink = np.where(a[yb, int(w * 0.6):] > 128)[0] + int(w * 0.6)
        if len(ink):
            arm_x = int(ink.mean())
            a[y0:y1 + 1, int(w * 0.62):] = 0
            a[y0:y1 + 3, arm_x - 3:arm_x + 4] = a[yb, arm_x - 3:arm_x + 4]
            img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
            r = h * 0.052; cx, cy = arm_x + r * 0.55, y0 + r * 0.2
            ImageDraw.Draw(img).ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
            A = Image.new('RGBA', img.size, GOLD + (0,)); A.putalpha(img)
    return A, top


def _glyph(text, font, color, track=0.0):
    """Texto en RGBA recortado a su caja de tinta; track = espaciado extra en fracción del tamaño."""
    size = font.size
    w = int(font.getlength(text) + track * size * len(text) + size)
    im = Image.new('L', (w + 40, int(size * 1.6)), 0)
    d = ImageDraw.Draw(im)
    x = 20
    for ch in text:
        d.text((x, int(size * 0.2)), ch, font=font, fill=255)
        x += font.getlength(ch) + track * size
    bbox = im.getbbox()
    a = im.crop(bbox)
    rgba = Image.new('RGBA', a.size, color + (0,)); rgba.putalpha(a)
    top = bbox[1] - int(size * 0.2)
    return rgba, top


def build(cap=160, serif='BodoniModa[opsz,wght].ttf', serif_w=600, sans='Jost[wght].ttf', sans_w=500, track=0.62):
    """Devuelve capas y posiciones relativas al centro superior del bloque (en píxeles)."""
    size = int(cap / 0.70)                        # alto de mayúscula ≈ 0.70 em
    fs = _font(serif, size, serif_w)
    S, _ = _glyph('S', fs, IVORY); A, _ = _ampersand(fs); P, _ = _glyph('P', fs, IVORY)
    gap = int(cap * 0.035)
    total = S.width + A.width + P.width + 2 * gap
    hS = max(S.height, P.height)
    x0 = -total // 2
    layers = {
        'S': (S, x0, hS - S.height),
        'amp': (A, x0 + S.width + gap, hS - A.height + int(cap * 0.01)),
        'P': (P, x0 + S.width + A.width + 2 * gap, hS - P.height),
    }
    line_w = int(cap * 1.05); line_h = max(2, int(round(cap * 0.035)))
    line = Image.new('RGBA', (line_w, line_h), GOLD + (255,))
    ly = hS + int(cap * 0.47)
    layers['line'] = (line, -line_w // 2, ly)
    bsize = int(cap * 0.285 / 0.70)
    fb = _font(sans, bsize, sans_w)
    B, _ = _glyph('BOUTIQUE', fb, IVORY, track=track)
    # ajusta el espaciado para que BOUTIQUE mida lo mismo que S&P
    for _ in range(6):
        ratio = total / B.width
        track = track + (ratio - 1) * 0.9
        B, _ = _glyph('BOUTIQUE', fb, IVORY, track=track)
    layers['boutique'] = (B, -B.width // 2, ly + int(cap * 0.37))
    layers['_track'] = track
    return layers


def flatten(layers, pad=20):
    items = [(k, v) for k, v in layers.items() if not k.startswith('_')]
    xs = [x for _, (im, x, y) in items] + [x + im.width for _, (im, x, y) in items]
    ys = [y for _, (im, x, y) in items] + [y + im.height for _, (im, x, y) in items]
    W, H = max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad
    canvas = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    for _, (im, x, y) in items:
        canvas.alpha_composite(im, (x - min(xs) + pad, y - min(ys) + pad))
    return canvas
