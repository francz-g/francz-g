"""Guion de montaje (EDL) del Reel: planos sincronizados a 'Slow Trance Morning' (124 BPM)."""
import json, os, sys
SP = os.environ.get('REEL_DIR', '.')
B = 60 / 124  # un tiempo a 124 BPM
FINAL_DUR = 4.32  # último plano: logo + cierre (total < 45 s)


def cam(**k):
    # estabilización fuerte por defecto (2 s): elimina el bamboleo de cámara en mano con poco recorte extra
    c = dict(stab=2.0, follow=0.5, head=0.07, zoom=1.0); c.update(k); return c


W = lambda d, f=8: dict(type='whip', dir=d, frames=f)
# (clip, entrada en s, tiempos de música, velocidad, cámara, transición de entrada, efectos)
shots = [
    # Apertura: look crema + satén chocolate. Cámara lenta en la intro y velocidad real justo en el "drop"
    ('IMG_6219', 5.05, 8, [[0, 0.5], [1.85, 0.5], [2.0, 1.0], [3.91, 1.0]], cam(push=0.05), None, None),
    # Look 1: trench crema + top estampado (entra en el drop con barrido siguiendo el giro)
    ('IMG_6181', 5.35, 4, 1.0, cam(), W('left'), None),
    ('IMG_6181', 9.40, 4, 1.0, cam(zoom=1.15, head=0.08), None, None),
    # Look 2: camisola gris satinada + pantalón blanco (frente y espalda descubierta)
    ('IMG_6185', 5.25, 4, 1.0, cam(stab=1.2), W('right'), None),
    ('IMG_6185', 9.10, 4, 0.8, cam(zoom=1.06), dict(type='soft', frames=4), None),
    # Look 3: denim (dos tomas)
    ('IMG_6189', 3.90, 4, 1.0, cam(push=0.04), dict(type='zoom', frames=8, amount=0.30), [dict(type='punch', t=2 * B, amount=0.045, frames=5)]),
    ('IMG_6188', 5.60, 4, 1.0, cam(zoom=1.05), None, None),
    # Look 9: maxi azul marino, giro de falda en cámara lenta
    ('IMG_6212', 7.45, 4, 1.0, cam(), W('up'), None),
    ('IMG_6212', 11.25, 4, 0.5, cam(), None, None),
    # Look 6: chaqueta de pelo bicolor
    ('IMG_6207', 5.75, 4, 1.0, cam(), W('left'), None),
    ('IMG_6207', 8.45, 2, 1.0, cam(zoom=1.2, head=0.09), None, None),
    # Look 4: plumífero negro con capucha de pelo
    ('IMG_6201', 5.50, 4, 1.0, cam(), dict(type='zoom', frames=8, amount=0.28), None),
    ('IMG_6201', 11.45, 2, 0.8, cam(), None, None),
    # Look 5: abrigo de pelo leopardo (apertura del abrigo)
    ('IMG_6202', 5.90, 4, 1.0, cam(), W('right'), None),
    ('IMG_6202', 12.95, 4, 0.75, cam(), None, None),
    # Look 8: kaftán leopardo (match de estampado con zoom) y giro en cámara lenta
    ('IMG_6211', 6.20, 4, 1.0, cam(), dict(type='zoom', frames=8, amount=0.35), None),
    # el operador acompaña el giro con la cámara: se respeta ese movimiento (estabilizar obligaría a cortar los pies)
    ('IMG_6211', 8.30, 4, 0.5, cam(stab=0, follow=0.3, head=0.05), None, None),
    # Look 10: vestido negro escote V (con chaqueta y solo)
    ('IMG_6217', 6.25, 3, 1.0, cam(), dict(type='blur', frames=8, sigma=12), None),
    ('IMG_6215', 11.00, 3, 1.0, cam(), dict(type='turn', dir='left', frames=6), None),
    # Look 7: vestido negro de espalda descubierta (revelación en cámara lenta)
    ('IMG_6208', 7.70, 2, 1.0, cam(), W('left'), None),
    ('IMG_6208', 8.65, 4, 0.6, cam(zoom=1.04, stab=1.2), None, None),
    ('IMG_6208', 11.15, 4, 0.8, cam(), None, None),
    # Cierre: vuelve el look de la paleta de la tienda y funde a chocolate
    # encuadre con aire arriba para que el logo quede sobre la pared, como en la referencia de la tienda
    ('IMG_6219', 16.70, None, 0.5, cam(follow=0.3, head=None, ycenter=0.46), dict(type='blur', frames=12, sigma=12), None),
]
out = []
for clip, tin, beats, sp, c, tr, fx in shots:
    d = {'clip': clip, 'in': tin, 'dur': round(beats * B, 5) if beats else FINAL_DUR, 'speed': sp, 'cam': c}
    if tr: d['trans_in'] = tr
    if fx: d['fx'] = fx
    out.append(d)
edl = dict(fps=30, size=[1080, 1920], probe=SP + '/work/probe.json', proxy_dir=SP + '/work/proxy',
           look=dict(lut=SP + '/work/boutique.cube', vignette=0.14, overlay=dict(dir=SP + '/work/logo_overlay')),
           music=dict(file=SP + '/work/music_bed_slowtrance.wav', start=0.0, fade_in=0.0, fade_out=0.3, gain_db=-6.3),
           # brillo sonoro sincronizado con la aparición del título "outfit inspo" (Mixkit 'Fairy glitter')
           sfx=[dict(file=SP + '/sfx/s867.mp3', t=0.05, gain_db=-2.0)], shots=out)
json.dump(edl, open(sys.argv[1], 'w'), indent=1)
print('planos', len(out), 'duración', round(sum(s['dur'] for s in out), 3), 's')
