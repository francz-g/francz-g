#!/usr/bin/env bash
# Proxies 540px de ancho (SDR BT.709) para análisis rápido. Uso: make_proxies.sh SRC DST
SRC=$1; DST=$2; mkdir -p "$DST"
one() {
  f=$1; DST=$2; n=$(basename "$f" .MOV)
  trc=$(ffprobe -v error -select_streams v:0 -show_entries stream=color_transfer -of csv=p=0 "$f")
  if [ "$trc" = "arib-std-b67" ] || [ "$trc" = "smpte2084" ]; then
    vf="zscale=w=540:h=960:f=spline36:t=linear:npl=203,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=mobius:param=0.35:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
  else
    vf="scale=540:-2:flags=bicubic,format=yuv420p"
  fi
  ffmpeg -nostdin -v error -y -i "$f" -vf "$vf" -c:v libx264 -preset veryfast -crf 19 -an "$DST/$n.mp4" && echo "proxy $n"
}
export -f one
ls "$SRC"/*.MOV | xargs -P 3 -I{} bash -c 'one "{}" "'"$DST"'"'
