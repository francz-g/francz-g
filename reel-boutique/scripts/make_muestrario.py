"""Muestrario musical: cada opción suena 12 s sobre el Reel, con su nombre en pantalla.
Uso: make_muestrario.py OPTIONS.json SCAN.json MUSIC_DIR REEL.mp4 OUT.mp4"""
import json, os, subprocess, sys
import numpy as np, librosa, soundfile as sf, warnings
from PIL import Image, ImageDraw, ImageFont
warnings.filterwarnings('ignore')
opts, scan = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
mdir, reel, out = sys.argv[3], sys.argv[4], sys.argv[5]
work = os.path.join(os.path.dirname(out), 'muestrario_tmp'); os.makedirs(work, exist_ok=True)
FONTS = os.path.join(os.path.dirname(__file__), '..', 'fonts')
SEG, SR, W, H = 12.0, 48000, 720, 1280
IVORY, GOLD, CHOC = (252, 251, 247), (221, 196, 145), (52, 33, 22)


def font(name, size, wght):
    f = ImageFont.truetype(os.path.join(FONTS, name), size)
    try:
        f.set_variation_by_axes([wght] + ([24] if 'opsz' in name else []))
    except Exception:
        pass
    return f


def spaced(d, xy, text, f, fill, track):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=f, fill=fill); x += f.getlength(ch) + track


def caption(o, path):
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 205), fill=CHOC + (205,))
    d.line((W / 2 - 40, 150, W / 2 + 40, 150), fill=GOLD + (255,), width=2)
    f1 = font('Jost[wght].ttf', 26, 500); f2 = font('BodoniModa[opsz,wght].ttf', 46, 600); f3 = font('Jost[wght].ttf', 24, 400)
    lab = f"OPCIÓN {o['n']}"; tw = sum(f1.getlength(c) + 6 for c in lab) - 6
    spaced(d, ((W - tw) / 2, 30), lab, f1, GOLD + (255,), 6)
    t = o['name']; d.text(((W - f2.getlength(t)) / 2, 66), t, font=f2, fill=IVORY + (255,))
    sub = f"{o['artist']}  ·  {o['style']}  ·  ~{o['bpm']} BPM"
    d.text(((W - f3.getlength(sub)) / 2, 164), sub, font=f3, fill=IVORY + (235,))
    im.save(path)


audio = []
for o in opts:
    s = scan[o['id']]
    y, _ = librosa.load(os.path.join(mdir, f"m{o['id']}.mp3"), sr=SR, mono=False)
    y = np.atleast_2d(y)
    if y.shape[0] == 1: y = np.vstack([y, y])
    start = max(0.0, min(s['excerpt'], s['dur'] - SEG - 0.5))
    seg = y[:, int(start * SR):int((start + SEG) * SR)].copy()
    if seg.shape[1] < int(SEG * SR):
        seg = np.pad(seg, ((0, 0), (0, int(SEG * SR) - seg.shape[1])))
    rms = np.sqrt(np.mean(seg ** 2)) + 1e-9
    seg *= (10 ** (-17 / 20)) / rms                      # mismo volumen percibido aproximado en todas
    fi, fo = int(0.25 * SR), int(0.7 * SR)
    seg[:, :fi] *= np.linspace(0, 1, fi); seg[:, -fo:] *= np.linspace(1, 0, fo) ** 2
    audio.append(np.clip(seg, -0.98, 0.98)); o['start_used'] = round(start, 2)
    caption(o, os.path.join(work, f"cap_{o['n']:02d}.png"))
sf.write(os.path.join(work, 'audio.wav'), np.concatenate(audio, 1).T, SR, subtype='PCM_16')
total = SEG * len(opts)
cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-stream_loop', '4', '-i', reel]
for o in opts:
    cmd += ['-i', os.path.join(work, f"cap_{o['n']:02d}.png")]
cmd += ['-i', os.path.join(work, 'audio.wav')]
fc = f"[0:v]scale={W}:{H}:flags=lanczos,trim=0:{total},setpts=PTS-STARTPTS[v0]"
for i, o in enumerate(opts):
    t0, t1 = i * SEG, (i + 1) * SEG
    fc += f";[v{i}][{i + 1}:v]overlay=0:0:enable='between(t,{t0},{t1 - 0.001})'[v{i + 1}]"
cmd += ['-filter_complex', fc, '-map', f'[v{len(opts)}]', '-map', f'{len(opts) + 1}:a',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
        '-t', str(total), '-movflags', '+faststart', out]
subprocess.run(cmd, check=True)
json.dump(opts, open(os.path.join(work, 'options_used.json'), 'w'), ensure_ascii=False, indent=1)
print('OK', out, total, 's')
