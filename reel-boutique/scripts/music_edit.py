"""Edición musical por compases: concatena segmentos del tema con microfundidos y cierre natural.
Uso: music_edit.py IN.mp3 OUT.wav "a0-a1,b0-b1,..." [xfade_s=0.025] [fade_out_s=0]"""
import sys, numpy as np, librosa, soundfile as sf, warnings
warnings.filterwarnings('ignore')
src, out, segs = sys.argv[1], sys.argv[2], sys.argv[3]
xf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.025
fo = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
y, sr = librosa.load(src, sr=48000, mono=False)
y = np.atleast_2d(y)
parts = []
for s in segs.split(','):
    a, b = map(float, s.split('-'))
    parts.append(y[:, int(a * sr):int(b * sr)])
n = int(xf * sr)
res = parts[0]
for p in parts[1:]:
    w = np.linspace(0, 1, n)
    fade_in, fade_out = np.sin(w * np.pi / 2), np.cos(w * np.pi / 2)
    mid = res[:, -n:] * fade_out + p[:, :n] * fade_in
    res = np.concatenate([res[:, :-n], mid, p[:, n:]], axis=1)
if fo > 0:
    k = int(fo * sr); res[:, -k:] *= np.linspace(1, 0, k) ** 2
sf.write(out, res.T, sr, subtype='PCM_24')
print(out, f"{res.shape[1] / sr:.3f}s")
