"""Rasgos de cada pista candidata: tempo, rejilla de tiempos, calidez, dinámica, intro y mejor tramo para el muestrario."""
import sys, glob, os, json, numpy as np, librosa, warnings
warnings.filterwarnings('ignore')
d, out = sys.argv[1], sys.argv[2]
meta = {m['id']: m for m in json.load(open(sys.argv[3]))} if len(sys.argv) > 3 else {}
res = {}
for f in sorted(glob.glob(os.path.join(d, 'm*.mp3'))):
    tid = os.path.basename(f)[1:-4]
    y, sr = librosa.load(f, sr=22050, mono=True)
    dur = len(y) / sr
    hop = 256
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=hop, tightness=200)
    tempo = float(np.atleast_1d(tempo)[0]); bt = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
    reg = float(np.std(np.diff(bt)) / np.mean(np.diff(bt))) if len(bt) > 4 else 9
    cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]; tt = librosa.times_like(rms, sr=sr, hop_length=hop)
    sec = np.array([rms[(tt >= s) & (tt < s + 1)].mean() for s in range(int(dur))])
    main = np.median(sec[int(len(sec) * 0.2):int(len(sec) * 0.8)])
    intro = next((s for s, v in enumerate(sec) if v >= 0.8 * main), 0)
    # mejor tramo de 12 s: energía alta y estable, empezando en un tiempo fuerte tras la intro
    best, bs = -1, intro
    for s in range(intro, max(intro + 1, int(dur) - 14)):
        seg = sec[s:s + 12]
        if len(seg) < 12: break
        score = seg.mean() / (main + 1e-9) - 0.5 * seg.std() / (seg.mean() + 1e-9)
        if score > best: best, bs = score, s
    bstart = float(bt[np.argmin(np.abs(bt - bs))]) if len(bt) else float(bs)
    h, p = librosa.effects.hpss(y)
    harm = float(np.sum(h ** 2) / (np.sum(h ** 2) + np.sum(p ** 2)))
    rms_db = 20 * np.log10(rms + 1e-9); dyn = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10))
    res[tid] = dict(dur=dur, bpm=tempo, reg=reg, cent=cent, harm=harm, dyn=dyn, intro=intro, excerpt=bstart,
                    name=meta.get(tid, {}).get('name', '?'), genre=meta.get(tid, {}).get('genre', '?'), artist=meta.get(tid, {}).get('artist', '?'))
    r = res[tid]
    print(f"{tid:>5} {r['name'][:26]:26} {r['genre'][:14]:14} bpm {tempo:6.1f} reg {reg:.3f} cent {cent:5.0f} harm {harm:.2f} dyn {dyn:4.1f} intro {intro:2d}s tramo {bstart:5.1f}s", flush=True)
json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
