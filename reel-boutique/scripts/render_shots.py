"""Pasada 1: cada plano del EDL -> clip intermedio 9:16 estabilizado, reencuadrado y normalizado.

Uso: render_shots.py EDL.json OUT_DIR [--preview] [--only i,j,...]
El EDL define fps, tamaño, y por plano: clip, in (s fuente), dur (s salida), speed, cam{}, gain[r,g,b], exposure,
trans_in{type, frames}. Los intermedios incluyen 'asas' antes/después para las transiciones.
"""
import json, os, sys, subprocess, math
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
from camera import read_proxy_frames, plan_camera

SP = os.environ.get('REEL_DIR', '.')


def handles(edl):
    shots = edl['shots']; res = []
    for i, s in enumerate(shots):
        pre = math.ceil(s.get('trans_in', {}).get('frames', 0) / 2) if i > 0 else 0
        nxt = shots[i + 1].get('trans_in', {}).get('frames', 0) if i + 1 < len(shots) else 0
        res.append((pre, math.ceil(nxt / 2)))
    return res


def frame_plan(edl):
    """Inicio (en fotogramas) de cada plano redondeando el tiempo acumulado, para no perder la sincronía con el ritmo."""
    fps = edl['fps']; acc = 0.0; starts = []
    for s in edl['shots']:
        starts.append(int(round(acc * fps))); acc += s['dur']
    starts.append(int(round(acc * fps)))
    return starts[:-1], [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]


def src_times(shot, fps, pre, post, n):
    """Tiempo de fuente para cada fotograma de salida k = -pre .. n+post-1 (rampas de velocidad incluidas)."""
    ks = np.arange(-pre, n + post)
    sp = shot.get('speed', 1.0)
    if isinstance(sp, (int, float)):
        return shot['in'] + ks / fps * sp, n
    kt, kv = np.array(sp, float).T  # [[t_salida, velocidad], ...]
    fine = np.arange(-pre - 1, n + post + 1) / fps
    v = np.interp(fine, kt, kv)
    cum = np.concatenate([[0], np.cumsum(v[1:] * np.diff(fine))])
    zero = np.interp(0.0, fine, cum)
    return shot['in'] + np.interp(ks / fps, fine, cum) - zero, n


def tonemap_chain(meta, width):
    if meta['transfer'] in ('arib-std-b67', 'smpte2084'):
        h = int(round(meta['h'] * width / meta['w'] / 2) * 2)
        return (f"zscale=w={width}:h={h}:f=spline36:t=linear:npl=203,format=gbrpf32le,zscale=p=bt709,"
                f"tonemap=tonemap=mobius:param=0.35:desat=0,zscale=t=bt709:m=bt709:r=tv,format=rgb24")
    return f"scale={width}:-2:flags=lanczos,format=rgb24"


class Decoder:
    """Lectura secuencial de fotogramas fuente [j0, j1] a la resolución de trabajo (sin cargar todo en memoria)."""

    def __init__(self, meta, j0, j1, width):
        self.w = width; self.h = int(round(meta['h'] * width / meta['w'] / 2) * 2)
        vf = tonemap_chain(meta, width)
        ss = max(0.0, (j0 - 0.5) / meta['fps'])
        self.p = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-ss', f"{ss:.5f}", '-i', meta['file'],
                                   '-frames:v', str(j1 - j0 + 1), '-vf', vf, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 8)
        self.idx = j0 - 1; self.cache = {}

    def get(self, j):
        """Fotograma j (j no decreciente salvo el anterior inmediato)."""
        while self.idx < j:
            buf = self.p.stdout.read(self.w * self.h * 3)
            if len(buf) < self.w * self.h * 3:
                break
            self.idx += 1
            self.cache[self.idx] = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
            for k in [k for k in self.cache if k < self.idx - 1]:
                del self.cache[k]
        return self.cache.get(j, self.cache[max(self.cache)])

    def close(self):
        self.p.kill(); self.p.stdout.close(); self.p.wait()


def flow_interp(a, b, alpha):
    """Fotograma intermedio entre a y b (RGB) por flujo óptico DIS."""
    s = 0.5
    ga = cv2.cvtColor(cv2.resize(a, None, fx=s, fy=s), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, None, fx=s, fy=s), cv2.COLOR_RGB2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fab = cv2.resize(dis.calc(ga, gb, None), (a.shape[1], a.shape[0])) / s
    fba = cv2.resize(dis.calc(gb, ga, None), (a.shape[1], a.shape[0])) / s
    h, w = a.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    t = alpha  # aproximación lineal de flujo intermedio (Super SloMo)
    ft0 = -(1 - t) * t * fab + t * t * fba
    ft1 = (1 - t) * (1 - t) * fab - t * (1 - t) * fba
    wa = cv2.remap(a, gx + ft0[..., 0], gy + ft0[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    wb = cv2.remap(b, gx + ft1[..., 0], gy + ft1[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.addWeighted(wa, 1 - t, wb, t, 0)


def main():
    edl = json.load(open(sys.argv[1])); odir = sys.argv[2]; os.makedirs(odir, exist_ok=True)
    preview = '--preview' in sys.argv
    only = None
    if '--only' in sys.argv:
        only = {int(x) for x in sys.argv[sys.argv.index('--only') + 1].split(',')}
    meta_all = json.load(open(edl.get('probe', SP + '/work/probe.json')))
    fps = edl['fps']; OW, OH = edl['size']
    if preview:
        OW, OH = OW // 2, OH // 2
    hs = handles(edl)
    _, ns = frame_plan(edl)
    log = {}
    for i, shot in enumerate(edl['shots']):
        if only is not None and i not in only:
            continue
        meta = meta_all[shot['clip']]
        pre, post = hs[i]
        n = ns[i]
        ts, _ = src_times(shot, fps, pre, post, n)
        sfps = meta['fps']
        jf = ts * sfps                                      # índice fraccional de fotograma fuente
        j0 = int(math.floor(jf.min())) - 1; j1 = int(math.ceil(jf.max())) + 1
        j0 = max(0, j0); j1 = min(j1, meta['nb_frames'] - 1 if meta['nb_frames'] else j1)
        # --- cámara (sobre proxy) ---
        pframes = read_proxy_frames(os.path.join(edl.get('proxy_dir', SP + '/work/proxy'), shot['clip'] + '.mp4'), j0, j1)
        mats, boxes, info = plan_camera(pframes, sfps, shot.get('cam', {}))
        PW = pframes[0].shape[1]
        # --- decodificación a resolución de trabajo ---
        work_w = 810 if preview else int(shot.get('work_w', 1620))
        dec = Decoder(meta, j0, j1, work_w)
        sc = PW / work_w
        S_in = np.diag([sc, sc, 1.0]); S_out = np.diag([OW, OH, 1.0])
        gain = np.array(shot.get('gain', [1, 1, 1]), np.float32) * (2 ** shot.get('exposure', 0.0))
        out = os.path.join(odir, f"shot_{i:02d}.mkv")
        enc = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f"{OW}x{OH}",
                                '-r', str(fps), '-i', '-', '-c:v', 'libx264rgb', '-preset', 'veryfast', '-crf', '4' if not preview else '14', out],
                               stdin=subprocess.PIPE)
        # con fuente a 60 fps el fotograma más cercano basta; 'flow' solo para cámaras lentas extremas
        interp = shot.get('interp', 'nearest')
        lut = None
        if not np.allclose(gain, 1):
            x = np.arange(256, dtype=np.float32) / 255
            lut = [np.clip(((x ** 2.2) * g) ** (1 / 2.2) * 255 + 0.5, 0, 255).astype(np.uint8) for g in gain]
        nf = j1 - j0 + 1
        for k, j in enumerate(jf):
            jl = j - j0
            if interp == 'nearest':
                jl = float(np.round(jl))
            a = int(np.clip(math.floor(jl), 0, nf - 1)); b = min(a + 1, nf - 1); al = float(jl - math.floor(jl))
            ma = min(a, len(mats) - 1); mb = min(b, len(mats) - 1)
            M = (1 - al) * mats[ma] + al * mats[mb]
            A = (S_out @ M @ S_in)[:2]
            fa = dec.get(j0 + a)
            if al < 0.02 or b == a:
                src = fa
            elif al > 0.98:
                src = dec.get(j0 + b)
            elif interp == 'flow':
                src = flow_interp(fa, dec.get(j0 + b), al)
            elif interp == 'blend':
                src = cv2.addWeighted(fa, 1 - al, dec.get(j0 + b), al, 0)
            else:
                src = fa if al < 0.5 else dec.get(j0 + b)
            o = cv2.warpAffine(src, A, (OW, OH), flags=cv2.INTER_LANCZOS4 if not preview else cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
            if lut is not None:
                o = np.dstack([cv2.LUT(o[..., c], lut[c]) for c in range(3)])
            enc.stdin.write(np.ascontiguousarray(o).tobytes())
        enc.stdin.close(); enc.wait(); dec.close()
        info.update(frames=len(jf), pre=pre, post=post, n=n, src_span=[float(ts.min()), float(ts.max())])
        log[i] = info
        print(f"shot {i:02d} {shot['clip']} {ts.min():.2f}-{ts.max():.2f}s  {len(jf)} fr  zoom {info['zoom_min']:.2f}-{info['zoom_max']:.2f}  pose {info['pose_frac']:.0%}", flush=True)
    json.dump(log, open(os.path.join(odir, 'render_log.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
