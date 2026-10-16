"""Verificación del Reel exportado: especificaciones, cortes vs. pulso musical y niveles."""
import sys, json, subprocess, numpy as np, cv2
f = sys.argv[1]; beat = 60 / float(sys.argv[2]) if len(sys.argv) > 2 else 60 / 123.0
j = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams', '-show_format', f],
                              capture_output=True, text=True).stdout)
for s in j['streams']:
    if s['codec_type'] == 'video':
        print(f"video: {s['codec_name']} {s.get('profile')} {s['width']}x{s['height']} {s['r_frame_rate']} fps {s['pix_fmt']} "
              f"{s.get('color_primaries')}/{s.get('color_transfer')}/{s.get('color_space')} {int(s.get('bit_rate', 0)) / 1e6:.1f} Mbps")
    else:
        print(f"audio: {s['codec_name']} {s['sample_rate']} Hz {s['channels']} ch {int(s.get('bit_rate', 0)) / 1e3:.0f} kbps")
print(f"duración {float(j['format']['duration']):.2f} s, tamaño {int(j['format']['size']) / 1e6:.1f} MB")
# cortes duros: picos de diferencia entre fotogramas consecutivos
cap = cv2.VideoCapture(f); fps = cap.get(cv2.CAP_PROP_FPS); prev = None; d = []
while True:
    ok, fr = cap.read()
    if not ok: break
    g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (90, 160)).astype(np.float32)
    if prev is not None: d.append(np.mean(np.abs(g - prev)))
    prev = g
d = np.array(d); thr = max(12.0, np.percentile(d, 97))
cuts = [i + 1 for i in range(1, len(d) - 1) if d[i] > thr and d[i] >= d[i - 1] and d[i] >= d[i + 1]]
offs = [((c / fps) / beat - round((c / fps) / beat)) * beat * 1000 for c in cuts]
print(f"cortes duros detectados: {len(cuts)}; desfase respecto al pulso (ms): {[round(o) for o in offs]}")
