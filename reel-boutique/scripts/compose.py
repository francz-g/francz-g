"""Pasada 2: une los planos intermedios con transiciones, aplica el look (LUT + viñeta) y exporta el Reel.

Uso: compose.py EDL.json SHOTS_DIR OUT.mp4 [--preview] [--no-music]
Transiciones (trans_in del plano B): cut, whip(dir), push(dir), zoom, blur, dip(color), soft, turn(dir)
Efectos dentro del plano (fx): punch{t, amount, frames}
"""
import json, math, os, subprocess, sys
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(__file__))
from render_shots import handles, frame_plan

SP = os.environ.get('REEL_DIR', '.')


class Reader:
    def __init__(self, path, w, h):
        self.p = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-i', path, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                                  stdout=subprocess.PIPE)
        self.sz = w * h * 3; self.w, self.h = w, h; self.last = None

    def next(self):
        buf = self.p.stdout.read(self.sz)
        if len(buf) == self.sz:
            self.last = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
        return self.last  # si se acaba, repite el último fotograma

    def close(self):
        self.p.stdout.close(); self.p.wait()


def ease_in(x): return x ** 3
def ease_out(x): return 1 - (1 - x) ** 3
def smooth(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def mblur(img, length, horizontal=True):
    L = int(round(abs(length)))
    if L < 2:
        return img
    return cv2.blur(img, (L, 1) if horizontal else (1, L), borderType=cv2.BORDER_REFLECT101)


def scale_about(img, s, cx=None, cy=None):
    h, w = img.shape[:2]; cx = w / 2 if cx is None else cx; cy = h / 2 if cy is None else cy
    M = np.float32([[s, 0, cx - s * cx], [0, s, cy - s * cy]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def zoom_blur(img, s0, s1, n=7):
    acc = np.zeros(img.shape, np.float32)
    for s in np.linspace(s0, s1, n):
        acc += scale_about(img, s)
    return (acc / n).astype(np.uint8)


DIRS = {'left': (-1, 0), 'right': (1, 0), 'up': (0, -1), 'down': (0, 1)}


def transition(kind, A, B, p, tr):
    """Mezcla A (sale) y B (entra) con progreso p en [0,1]."""
    h, w = A.shape[:2]
    if kind in ('whip', 'turn'):
        # barrido: A acelera y se desenfoca en la dirección del movimiento; B entra desde el lado opuesto y frena
        dx, dy = DIRS[tr.get('dir', 'left')]
        size = w if dx else h
        span = size * tr.get('amp', 0.42 if kind == 'whip' else 0.10)
        blur_k = size * tr.get('blur', 0.16 if kind == 'whip' else 0.05)

        def a_img(p):
            q = min(1.0, p / 0.5); off = ease_in(q) * span
            return mblur(shift(A, dx * off, dy * off), blur_k * 3 * q * q, horizontal=bool(dx))

        def b_img(p):
            q = max(0.0, (p - 0.5) / 0.5); off = -(1 - ease_out(q)) * span
            return mblur(shift(B, dx * off, dy * off), blur_k * 3 * (1 - q) ** 2, horizontal=bool(dx))

        m = smooth((p - 0.44) / 0.12)
        if m <= 0:
            return a_img(p)
        if m >= 1:
            return b_img(p)
        return cv2.addWeighted(a_img(p), 1 - m, b_img(p), m, 0)
    if kind == 'push':
        # B empuja a A fuera de cuadro (deslizamiento suave con desenfoque de movimiento)
        dx, dy = DIRS[tr.get('dir', 'left')]
        e = smooth(p); size = w if dx else h
        k = tr.get('blur', 0.10) * size * 6 * p * (1 - p)
        a = mblur(shift(A, dx * e * size, dy * e * size), k, bool(dx))
        b = mblur(shift(B, dx * (e - 1) * size, dy * (e - 1) * size), k, bool(dx))
        mask = np.zeros((h, w), np.float32)
        cut = int(round(e * size))
        if dx > 0: mask[:, :cut] = 1
        elif dx < 0: mask[:, size - cut:] = 1
        elif dy > 0: mask[:cut] = 1
        else: mask[size - cut:] = 1
        mask = cv2.GaussianBlur(mask, (0, 0), 4)[..., None]
        return (a * (1 - mask) + b * mask).astype(np.uint8)
    if kind == 'zoom':
        # zoom a través: A se acerca con desenfoque radial creciente; B sale del mismo zoom y se asienta
        z = tr.get('amount', 0.35); k = tr.get('blur', 0.20)

        def a_img(p):
            q = min(1.0, p / 0.5); s = 1 + z * ease_in(q)
            return zoom_blur(A, s, s * (1 + k * q * q), 9)

        def b_img(p):
            q = max(0.0, (p - 0.5) / 0.5); s = 1 + z * (1 - ease_out(q))
            return zoom_blur(B, s, s * (1 + k * (1 - q) ** 2), 9)

        m = smooth((p - 0.42) / 0.16)
        if m <= 0:
            return a_img(p)
        if m >= 1:
            return b_img(p)
        return cv2.addWeighted(a_img(p), 1 - m, b_img(p), m, 0)
    if kind == 'blur':
        sig = tr.get('sigma', 14) * math.sin(math.pi * p) + 0.01
        m = smooth(p)
        a = cv2.GaussianBlur(A, (0, 0), sig); b = cv2.GaussianBlur(B, (0, 0), sig)
        return cv2.addWeighted(a, 1 - m, b, m, 0)
    if kind == 'dip':
        col = np.array(tr.get('color', [243, 235, 221]), np.uint8)
        plate = np.empty_like(A); plate[:] = col
        if p < 0.5:
            return cv2.addWeighted(A, 1 - smooth(p / 0.5), plate, smooth(p / 0.5), 0)
        return cv2.addWeighted(plate, 1 - smooth((p - 0.5) / 0.5), B, smooth((p - 0.5) / 0.5), 0)
    if kind == 'soft':
        return cv2.addWeighted(A, 1 - smooth(p), B, smooth(p), 0)
    return B if p >= 0.5 else A


def apply_fx(img, fxs, t_in_shot, fps):
    for fx in fxs or []:
        if fx['type'] == 'punch':
            d = fx.get('frames', 8) / fps; t0 = fx['t']
            if t0 <= t_in_shot < t0 + d * 3:
                q = (t_in_shot - t0) / d
                amt = fx.get('amount', 0.07) * (ease_out(min(q, 1)) if q < 1 else max(0, 1 - smooth((q - 1) / 2)) if fx.get('release', True) else 1)
                img = scale_about(img, 1 + amt)
        if fx['type'] == 'hold_zoom':  # zoom que se queda (sin retorno)
            d = fx.get('frames', 8) / fps; t0 = fx['t']
            if t_in_shot >= t0:
                q = min(1, (t_in_shot - t0) / d)
                img = scale_about(img, 1 + fx.get('amount', 0.07) * ease_out(q))
    return img


def main():
    edl = json.load(open(sys.argv[1])); sdir = sys.argv[2]; out = sys.argv[3]
    preview = '--preview' in sys.argv; nomusic = '--no-music' in sys.argv
    fps = edl['fps']; W, H = edl['size']
    if preview:
        W, H = W // 2, H // 2
    shots = edl['shots']; hs = handles(edl)
    starts, ns = frame_plan(edl)
    total = starts[-1] + ns[-1]
    # viñeta sutil y cálida
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
    vig_amt = edl.get('look', {}).get('vignette', 0.16)
    vig = (1 - vig_amt * np.clip((r - 0.35) / 0.65, 0, 1) ** 1.8)[..., None]
    vig = np.concatenate([vig * 1.0, vig * 0.995, vig * 0.985], axis=2) if vig_amt > 0 else None
    look = edl.get('look', {})
    vf = []
    if look.get('lut'):
        vf.append(f"lut3d=file={look['lut']}:interp=tetrahedral")
    if look.get('grain', 0) > 0:
        vf.append(f"noise=c0s={look['grain']}:c0f=t+u")
    cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-']
    ov = look.get('overlay')                       # secuencia PNG RGBA (logo y cierre) que va DESPUÉS del LUT
    if ov:
        cmd += ['-framerate', str(fps), '-i', os.path.join(ov['dir'], 'logo_%04d.png')]
    mus = edl.get('music')
    use_music = bool(mus) and not nomusic
    if use_music:
        cmd += ['-ss', f"{mus['start']:.3f}", '-i', mus['file']]
    base = ','.join(vf) if vf else 'null'
    if ov:
        scale = '' if not preview else f",scale={W}:{H}"
        fc = f"[0:v]{base}[b];[1:v]format=rgba{scale}[o];[b][o]overlay=0:0:format=auto:eof_action=pass,format=yuv420p[v]"
    else:
        fc = f"[0:v]{base},format=yuv420p[v]"
    cmd += ['-filter_complex', fc, '-map', '[v]',
            '-c:v', 'libx264', '-profile:v', 'high', '-preset', 'veryfast' if preview else 'slow',
            '-crf', '22' if preview else '16', '-maxrate', '20M', '-bufsize', '40M', '-g', str(int(fps * 2)),
            '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', '-color_range', 'tv']
    if use_music:
        dur = total / fps
        fo = mus.get('fade_out', 1.5)
        af = (f"atrim=0:{dur:.3f},afade=t=in:st=0:d={mus.get('fade_in', 0.05)},afade=t=out:st={dur - fo:.3f}:d={fo},"
              f"volume={mus.get('gain_db', 0)}dB,aresample=48000")  # ganancia fija medida (-14 LUFS), sin compresión
        cmd += ['-map', f"{2 if ov else 1}:a", '-af', af, '-c:a', 'aac', '-b:a', '256k', '-ar', '48000', '-ac', '2', '-shortest']
    cmd += ['-movflags', '+faststart', out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    readers = {}

    def frame_of(i, T):
        """Fotograma del plano i en el tiempo de línea T (consumo secuencial)."""
        pre, post = hs[i]
        f = T - (starts[i] - pre)
        rd = readers.get(i)
        if rd is None:
            rd = readers[i] = Reader(os.path.join(sdir, f'shot_{i:02d}.mkv'), W, H); rd.pos = -1
        while rd.pos < f:
            rd.next(); rd.pos += 1
        img = rd.last
        return apply_fx(img, shots[i].get('fx'), (T - starts[i]) / fps, fps)

    for T in range(total):
        # plano principal
        i = max(k for k in range(len(shots)) if starts[k] <= T)
        img = None
        # ¿estamos en una transición hacia el plano i (inicio) o hacia el i+1 (final)?
        for b in (i, i + 1):
            if 0 < b < len(shots):
                tr = shots[b].get('trans_in', {}); D = tr.get('frames', 0)
                if D > 0 and starts[b] - math.ceil(D / 2) <= T < starts[b] - math.ceil(D / 2) + D:
                    p = (T - (starts[b] - math.ceil(D / 2)) + 0.5) / D
                    img = transition(tr['type'], frame_of(b - 1, T), frame_of(b, T), p, tr)
                    break
        if img is None:
            img = frame_of(i, T)
        for k in [k for k in readers if k < i - 1]:
            readers.pop(k).close()
        if vig is not None:
            img = np.clip(img.astype(np.float32) * vig, 0, 255).astype(np.uint8)
        ef = look.get('end_fade')
        if ef and T >= total - ef['frames']:
            m = smooth((T - (total - ef['frames']) + 1) / ef['frames'])
            plate = np.empty_like(img); plate[:] = np.array(ef['color'], np.uint8)
            img = cv2.addWeighted(img, 1 - m, plate, m, 0)
        enc.stdin.write(np.ascontiguousarray(img).tobytes())
    enc.stdin.close(); enc.wait()
    for rd in readers.values():
        rd.close()
    print('OK', out, f'{total / fps:.2f}s', total, 'frames')


if __name__ == '__main__':
    main()
