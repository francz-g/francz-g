# Reel de moda para boutique: proyecto de edición

Reel vertical para Instagram (9:16, 1080×1920, H.264, 30 fps, 44,6 s), montado a partir de 13 tomas
grabadas con iPhone (4K HDR/HLG a 60 fps) de una modelo mostrando 11 looks.
Este directorio guarda el **proceso de edición reproducible**. No incluye los videos de origen
ni el Reel exportado, porque este repositorio es público.

## Dirección creativa

- **Paleta:** chocolate, beige cálido, crema/marfil y camel. Se aplica una LUT propia
  (`boutique.cube`, generada con `scripts/make_lut.py`) que añade:
  - sombras cálidas suaves y altas luces cremosas con *roll-off*;
  - contraste moderado;
  - saturación global −4 %, protegiendo los tonos de piel.

  La intención es no alterar el color real de las prendas.
- **HDR → SDR:** las tomas HLG BT.2020 se convierten a BT.709 con `zscale` + `tonemap=mobius`
  (pico nominal de 203 nits). Así se conserva la textura de las prendas crema y blancas.
- **Estructura:** la apertura y el cierre usan el look crema + satén chocolate (la paleta de la tienda).
  En medio aparecen los 11 looks, ordenados por color: neutros → denim → azul → bicolor →
  abrigos → leopardo → negro de noche.
- **Ritmo:** cortes sobre el pulso de *House 02* (123 BPM), casi siempre de un compás (1,95 s).
  Hay acentos de medio compás y cámaras lentas reales (fuente a 60 fps) en giros y vuelos de falda.
- **Estabilización:** el iPhone ya elimina la vibración rápida (unos 0,2 px).
  Lo que se corrige es el bamboleo de cámara en mano (periodos de 0,2 a 1 s), medido solo sobre el fondo,
  que baja de 3,5 a 1,9 px de media (−46 %) y hasta −70 % en las tomas más inestables.
  Se suaviza la trayectoria con σ = 2 s (1,2 s en dos tomas) y se usa un zoom constante por plano
  para no mostrar bordes. En el giro del kaftán se respeta el movimiento natural del operador:
  estabilizarlo habría obligado a cortar el vuelo del vestido.

## Guion de montaje (ver `edl.json`)

| # | Inicio | Look | Toma | Recurso |
|---|-------:|------|------|---------|
| 0 | 0,00 s | Cárdigan crema + pantalón satén chocolate | IMG_6219 | Cámara lenta 0,5× en la intro; velocidad real justo en el *drop* (1,95 s) |
| 1 | 3,90 s | Trench crema + top estampado | IMG_6181 | Barrido (whip pan) siguiendo el giro |
| 2 | 5,87 s | 〃 | IMG_6181 | Corte a plano más cerrado |
| 3 | 7,80 s | Camisola gris satinada + pantalón blanco | IMG_6185 | Barrido a la derecha |
| 4 | 9,77 s | 〃 (espalda) | IMG_6185 | Fundido corto, 0,8× |
| 5 | 11,70 s | Denim | IMG_6189 | Zoom de transición + *punch-in* en el pulso |
| 6 | 13,67 s | 〃 | IMG_6188 | Corte al pulso |
| 7 | 15,60 s | Maxi azul marino | IMG_6212 | Barrido vertical |
| 8 | 17,57 s | 〃 (vuelo de falda) | IMG_6212 | Cámara lenta 0,5× |
| 9 | 19,50 s | Chaqueta de pelo bicolor | IMG_6207 | Barrido |
| 10 | 21,47 s | 〃 | IMG_6207 | Plano medio, medio compás |
| 11 | 22,43 s | Plumífero negro con capucha | IMG_6201 | Zoom de transición |
| 12 | 24,40 s | 〃 (espalda) | IMG_6201 | 0,8× |
| 13 | 25,37 s | Abrigo de pelo leopardo | IMG_6202 | Barrido |
| 14 | 27,33 s | 〃 (abre el abrigo) | IMG_6202 | 0,75× |
| 15 | 29,27 s | Kaftán leopardo | IMG_6211 | *Match* de estampado con zoom |
| 16 | 31,23 s | 〃 (giro) | IMG_6211 | Cámara lenta 0,5× |
| 17 | 33,17 s | Vestido negro escote V + chaqueta | IMG_6217 | Fundido con desenfoque |
| 18 | 34,63 s | Vestido negro escote V | IMG_6215 | Transición de giro |
| 19 | 36,10 s | Vestido negro espalda descubierta | IMG_6208 | Barrido |
| 20 | 37,07 s | 〃 (revelación de la espalda) | IMG_6208 | Cámara lenta 0,6× |
| 21 | 39,03 s | 〃 | IMG_6208 | 0,8× |
| 22 | 40,97 s | Cierre: look crema + chocolate | IMG_6219 | Fundido con desenfoque, cámara lenta y fundido final a chocolate |

Se descartaron los primeros ~3 s de cada toma (sala vacía antes de que la modelo salga del biombo)
y los tramos repetitivos o con movimientos bruscos de cámara.

## Música

Pista usada: **"House 02" — Lily J** (Mixkit). Es deep house elegante a 123 BPM.
Está editada por compases: intro, cuerpo y el final natural del tema (`scripts/music_edit.py`).
Se distribuye bajo la *Mixkit Stock Music Free License*, que permite uso comercial y en redes sociales
sin atribución. No permite redistribuirla sola ni registrarla en sistemas de Content ID.

Alternativas recomendadas (misma licencia):

1. **"Lies and More Lies" — Michael Ramir C.** (disco, ~100 BPM): más retro-glam y seguro de sí mismo.
2. **"You're like Caviar" — Michael Ramir C.** (chillout / nu-jazz, ~112 BPM): lounge lujoso y relajado.
3. **"House 02" — Lily J**: la pista usada.

Ojo: las cuentas de empresa de Instagram solo pueden añadir audios de la *Meta Sound Collection*
(libres de derechos). Las bibliotecas de pago (Epidemic Sound, Artlist) incluyen licencia comercial para redes.

## Cómo reproducir

Requisitos: ffmpeg (con `zscale`/libzimg y `libx264`) y Python 3 con `opencv-python-headless`, `mediapipe`,
`librosa`, `scipy` y `matplotlib`. Para MediaPipe hacen falta `libegl1` y `libgles2`.

```bash
export REEL_DIR=/ruta/de/trabajo     # contiene src/ (videos .MOV), models/ (modelos de MediaPipe) y work/
python3 scripts/probe.py   $REEL_DIR/src $REEL_DIR/work/probe.json
scripts/make_proxies.sh     $REEL_DIR/src $REEL_DIR/work/proxy        # proxies 540 px, ya en SDR
python3 scripts/analyze.py  <proxy.mp4> <salida.npz> 6               # pose, giros, sonrisa, temblor
python3 scripts/make_lut.py $REEL_DIR/work/boutique.cube
python3 scripts/music_edit.py tema.mp3 $REEL_DIR/work/music_bed.wav "5.906-46.880,101.514-105.114" 0.02 1.6
python3 scripts/build_edl.py $REEL_DIR/work/edl.json
python3 scripts/render_shots.py $REEL_DIR/work/edl.json $REEL_DIR/work/shots   # estabiliza y reencuadra cada plano
python3 scripts/compose.py      $REEL_DIR/work/edl.json $REEL_DIR/work/shots reel.mp4   # transiciones, LUT, música
```

- `camera.py`: estabilización con el fondo (enmascarando a la modelo) y reencuadre 9:16 que sigue a la modelo con zoom constante.
- `compose.py`: transiciones de barrido, giro, zoom, fundido con desenfoque y fundido a color.
