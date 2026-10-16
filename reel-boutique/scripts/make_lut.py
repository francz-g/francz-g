"""LUT 3D (.cube) del look de la boutique: cálido y sofisticado sin alterar el color real de las prendas.
- curva S suave con negros levemente levantados (chocolate) y altas luces con roll-off (crema/marfil)
- split-toning muy sutil: sombras hacia marrón cálido, luces hacia crema
- saturación global -4 %, protegiendo tonos piel; sin dominante naranja."""
import sys, numpy as np
N = 33; out = sys.argv[1]
strength = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
g = np.linspace(0, 1, N)
b, gg, r = np.meshgrid(g, g, g, indexing='ij')          # orden .cube: R varía más rápido
rgb = np.stack([r, gg, b], -1).reshape(-1, 3)
def srgb_to_lin(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def lin_to_srgb(c): return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)
x = rgb.copy()
# 1) curva tonal (en gamma): S suave + toe/shoulder
def curve(v):
    toe = 0.018                                  # negros apenas levantados (sombras chocolate suaves)
    v2 = toe + (1 - toe - 0.02) * v              # techo 0.98 -> altas luces cremosas
    s = 0.5 + (v2 - 0.5) * 1.10                  # contraste moderado
    s = s + 0.035 * np.sin(np.pi * (v2 - 0.5) * 2) * (v2 > 0.5) * -0.3
    return np.clip(s, 0, 1)
x = curve(x)
# 2) split toning por luminancia
Y = 0.2126 * x[:, 0] + 0.7152 * x[:, 1] + 0.0722 * x[:, 2]
w_sh = np.clip(1 - Y / 0.45, 0, 1) ** 1.5; w_hi = np.clip((Y - 0.55) / 0.45, 0, 1) ** 1.2
shadow_tint = np.array([0.020, 0.006, -0.018])  # chocolate
hi_tint = np.array([0.008, 0.004, -0.008])      # crema/marfil (suave: los blancos siguen siendo blancos)
x = x + w_sh[:, None] * shadow_tint + w_hi[:, None] * hi_tint
# 3) saturación global con protección de piel (tonos 10-50° de hue)
mx, mn = x.max(1), x.min(1); Y = 0.2126 * x[:, 0] + 0.7152 * x[:, 1] + 0.0722 * x[:, 2]
hue = (np.degrees(np.arctan2(np.sqrt(3) * (x[:, 1] - x[:, 2]), 2 * x[:, 0] - x[:, 1] - x[:, 2])) + 360) % 360
skin = np.exp(-((hue - 28) / 18) ** 2) * np.clip((mx - mn) / 0.35, 0, 1)
sat = 0.96 + 0.04 * skin
x = Y[:, None] + (x - Y[:, None]) * sat[:, None]
x = np.clip(rgb + (x - rgb) * strength, 0, 1)
with open(out, 'w') as fh:
    fh.write('TITLE "Boutique warm"\nLUT_3D_SIZE 33\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
    for v in x: fh.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
print('LUT ok', out)
