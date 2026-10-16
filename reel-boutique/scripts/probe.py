"""Lee metadatos de cada clip (resolución, rotación, fps, duración, HDR)."""
import json, subprocess, sys, glob, os
src, out = sys.argv[1], sys.argv[2]
res = {}
for f in sorted(glob.glob(os.path.join(src, '*.MOV'))):
    j = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams', '-show_format', f],
                                  capture_output=True, text=True).stdout)
    v = next(s for s in j['streams'] if s['codec_type'] == 'video')
    rot = 0
    for sd in v.get('side_data_list', []):
        if 'rotation' in sd: rot = int(sd['rotation'])
    num, den = map(int, v['r_frame_rate'].split('/'))
    anum, aden = map(int, v['avg_frame_rate'].split('/'))
    w, h = v['width'], v['height']
    if abs(rot) == 90: w, h = h, w
    res[os.path.basename(f).split('.')[0]] = dict(
        file=f, w=w, h=h, rotation=rot, fps=num/den, avg_fps=anum/max(aden, 1), duration=float(j['format']['duration']),
        codec=v['codec_name'], pix_fmt=v.get('pix_fmt'), transfer=v.get('color_transfer'), primaries=v.get('color_primaries'),
        matrix=v.get('color_space'), nb_frames=int(v.get('nb_frames', 0)), bitrate=int(j['format'].get('bit_rate', 0)),
        has_audio=any(s['codec_type'] == 'audio' for s in j['streams']),
        creation=j['format'].get('tags', {}).get('creation_time'))
json.dump(res, open(out, 'w'), indent=1)
for k, r in res.items():
    print(f"{k}: {r['w']}x{r['h']} rot={r['rotation']} {r['fps']:.3f}fps {r['duration']:.1f}s {r['codec']} {r['pix_fmt']} trc={r['transfer']} prim={r['primaries']} {r['bitrate']/1e6:.0f}Mbps {r['creation']}")
print('TOTAL', round(sum(r['duration'] for r in res.values()), 1), 's')
