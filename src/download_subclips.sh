#!/bin/bash
# Descarga los subclips etiquetados (background / challenge / play / throwin)
# del dataset ManuD/dfl_clips de Hugging Face hacia data/subclips/<split>/<clase>/.
# Son recortes de los videos anotados en train.csv (DFL - Bundesliga Data Shootout).
# train: 10 videos de 7 partidos; valid: partido 3c993bd2 completo (~4.6 GB en total).
set -euo pipefail

DEST="data/subclips"
BASE="https://huggingface.co/datasets/ManuD/dfl_clips/resolve/main"
API="https://huggingface.co/api/datasets/ManuD/dfl_clips/tree/main/subclips"
LIST="$(mktemp)"

for split in train valid; do
  for cls in background challenge play throwin; do
    curl -s --max-time 60 "$API/$split/$cls" \
      | python3 -c "import sys,json; print('\n'.join(f['path'] for f in json.load(sys.stdin) if f['path'].endswith('.mp4')))" \
      >> "$LIST"
  done
done

export DEST BASE
fetch_one() {
  local path="$1" rel out
  rel="${path#subclips/}"; out="$DEST/$rel"
  mkdir -p "$(dirname "$out")"
  if [ -s "$out" ]; then return 0; fi
  curl -s -L --retry 4 --retry-delay 2 --max-time 180 --speed-limit 20000 --speed-time 30 "$BASE/$path" -o "$out.part" && mv "$out.part" "$out"
}
export -f fetch_one

xargs -P 8 -n 1 -I{} bash -c 'fetch_one "$@"' _ {} < "$LIST"
echo "Descargados $(find "$DEST" -name '*.mp4' | wc -l | tr -d ' ') subclips en $DEST"
