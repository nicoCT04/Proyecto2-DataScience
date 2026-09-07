#!/bin/bash
# Descarga los 200 clips de la Bundesliga desde el dataset de Hugging Face
# hacia data/clips/. Los videos no se versionan en git (~3.8 GB); este script
# los regenera. Requiere curl.
set -euo pipefail

DEST="data/clips"
BASE="https://huggingface.co/datasets/dbal0503/Bundesliga/resolve/main"
API="https://huggingface.co/api/datasets/dbal0503/Bundesliga/tree/main/Bundesliga/Clips?recursive=true"
mkdir -p "$DEST"

# Lista de rutas de los .mp4 en el dataset
curl -s --max-time 60 "$API" \
  | python3 -c "import sys,json; print('\n'.join(f['path'] for f in json.load(sys.stdin) if f['path'].endswith('.mp4')))" \
  > /tmp/clip_paths.txt

export DEST BASE
fetch_one() {
  local path="$1" name out
  name="$(basename "$path")"; out="$DEST/$name"
  if [ -f "$out" ] && [ "$(stat -f%z "$out")" -gt 1000000 ]; then return 0; fi
  curl -s -L --retry 4 --retry-delay 2 --max-time 300 "$BASE/$path" -o "$out"
}
export -f fetch_one

xargs -P 6 -n 1 -I{} bash -c 'fetch_one "$@"' _ {} < /tmp/clip_paths.txt
echo "Descargados $(ls "$DEST"/*.mp4 | wc -l | tr -d ' ') clips en $DEST"
