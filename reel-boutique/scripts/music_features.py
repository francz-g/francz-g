import sys, glob, os, numpy as np, librosa, warnings
warnings.filterwarnings('ignore')
d = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(d, 'm*.mp3'))):
    y, sr = librosa.load(f, sr=22050, mono=True)
    dur = len(y)/sr
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rms = librosa.feature.rms(y=y)[0]
    rms_db = 20*np.log10(rms+1e-9)
    dyn = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10))
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    perc = float(np.mean(onset))
    h, p = librosa.effects.hpss(y)
    harm_ratio = float(np.sum(h**2)/(np.sum(h**2)+np.sum(p**2)))
    # beat regularity
    bt = librosa.frames_to_time(beats, sr=sr)
    ibi = np.diff(bt); reg = float(np.std(ibi)/np.mean(ibi)) if len(ibi)>2 else 9
    rows.append((os.path.basename(f), dur, tempo, cent, dyn, perc, harm_ratio, reg))
print(f"{'file':10} {'dur':>5} {'bpm':>6} {'centroid':>8} {'dyn':>5} {'onset':>5} {'harm':>5} {'reg':>5}")
for r in sorted(rows, key=lambda r: r[3]):
    print(f"{r[0]:10} {r[1]:5.0f} {r[2]:6.1f} {r[3]:8.0f} {r[4]:5.1f} {r[5]:5.2f} {r[6]:5.2f} {r[7]:5.3f}")
