"""Resumen y gráficos de análisis por clip. Uso: summarize.py ANALYSIS_DIR OUT_DIR"""
import sys, os, glob, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
adir, odir = sys.argv[1], sys.argv[2]; os.makedirs(odir, exist_ok=True)
def nanfill(a):
    a = a.copy(); idx = np.arange(len(a)); ok = ~np.isnan(a)
    if ok.sum() == 0: return np.zeros_like(a)
    a[~ok] = np.interp(idx[~ok], idx[ok], a[ok]); return a
rows = []
for f in sorted(glob.glob(os.path.join(adir, '*.npz'))):
    d = np.load(f); name = os.path.basename(f)[:-4]; t = d['t']; sr = d['fps'] / d['step']
    # trayectoria de cámara (en fracción del ancho) y temblor = alta frecuencia
    dx, dy = nanfill(d['cam_dx']), nanfill(d['cam_dy'])
    px, py = np.cumsum(dx), np.cumsum(dy)
    k = max(3, int(sr * 1.0))
    shake = np.hypot(px - uniform_filter1d(px, k), py - uniform_filter1d(py, k)) * 100
    pan = np.hypot(uniform_filter1d(dx, k), uniform_filter1d(dy, k)) * sr * 100  # %ancho/s
    facing = nanfill(d['facing']); smile = np.nan_to_num(d['smile']); feet = np.nan_to_num(d['feet'])
    # giros: cambios de signo de 'facing' (frente <-> espalda)
    fs = uniform_filter1d(facing, 3); sign = np.sign(fs); sign[np.abs(fs) < 0.25] = 0
    turns = []; last = 0
    for i, s in enumerate(sign):
        if s != 0:
            if last != 0 and s != last: turns.append(t[i])
            last = s
    body_h = nanfill(d['y1'] - d['y0'])
    fig, ax = plt.subplots(5, 1, figsize=(11, 7.5), sharex=True)
    ax[0].plot(t, facing, c='#6b4423'); ax[0].axhline(0, c='gray', lw=0.5); ax[0].set_ylabel('frente(+)\nespalda(-)')
    for tt in turns: [a.axvline(tt, c='#c9a27e', lw=0.8, ls='--') for a in ax]
    ax[1].plot(t, smile, c='#b5651d'); ax[1].plot(t, np.nan_to_num(d['has_face']) * 0.05, c='gray', lw=0.5); ax[1].set_ylabel('sonrisa'); ax[1].set_ylim(0, 1)
    ax[2].fill_between(t, 1 - nanfill(d['y1']), 1 - nanfill(d['y0']), color='#d2b48c', alpha=0.6); ax[2].plot(t, feet, c='#4a2c17', lw=0.8)
    ax[2].set_ylabel('cuerpo en\ncuadro / pies'); ax[2].set_ylim(0, 1.05)
    ax[3].plot(t, shake, c='#8b0000', lw=0.8); ax[3].plot(t, pan / 10, c='#555', lw=0.8); ax[3].set_ylabel('temblor %\n(paneo/10)')
    ax[4].plot(t, np.nan_to_num(d['subj_motion']) * 100, c='#2f4f4f', lw=0.8); ax[4].plot(t, nanfill(d['sharp']) / max(1, np.nanmax(d['sharp'])) * 5, c='#999', lw=0.8)
    ax[4].set_ylabel('mov. modelo /\nnitidez'); ax[4].set_xlabel('s')
    fig.suptitle(f"{name}  ({t[-1]:.1f}s)  giros en: {', '.join(f'{x:.1f}' for x in turns)}", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(odir, name + '.png'), dpi=72); plt.close(fig)
    rows.append((name, t[-1], np.mean(d['has_pose']), np.nanmean(shake), np.nanpercentile(shake, 90), len(turns),
                 np.nanmax(smile) if len(smile) else 0, np.mean(feet > 0.5), np.nanmedian(body_h), np.nanmean(d['luma'])))
print(f"{'clip':9} {'dur':>5} {'pose%':>5} {'shake':>5} {'sh90':>5} {'turns':>5} {'smile':>5} {'feet%':>5} {'bodyH':>5} {'luma':>5}")
for r in rows: print(f"{r[0]:9} {r[1]:5.1f} {r[2]*100:5.0f} {r[3]:5.2f} {r[4]:5.2f} {r[5]:5d} {r[6]:5.2f} {r[7]*100:5.0f} {r[8]:5.2f} {r[9]:5.2f}")
