#!/usr/bin/env bash
# Host-side helper (needs Info-ZIP `zip` and `unzip`). Joins the split ImageCAS archives
# (<name>.change2zip is the last part, <name>.z01.. the others) and extracts each set into its
# own directory <raw>/<name>/. The original parts are kept; temporary files are removed on success.
set -euo pipefail
RAW="${TWINLOOP_DATA:-$HOME/twinloop-data}/raw/imagecas"
shopt -s nullglob
for last in "$RAW"/*.change2zip; do
  name="$(basename "$last" .change2zip)"
  out="$RAW/$name"
  if [ -e "$out/.extracted" ]; then echo "skip $name (already extracted)"; continue; fi
  echo "joining $name"
  ln -sfn "$last" "$RAW/$name.zip"
  zip -q -s 0 "$RAW/$name.zip" --out "$RAW/$name.joined.zip"
  rm -- "$RAW/$name.zip"
  mkdir -p "$out"
  unzip -q -o "$RAW/$name.joined.zip" -d "$out"
  touch "$out/.extracted"
  rm -- "$RAW/$name.joined.zip"
  echo "extracted $name"
done
echo JOIN DONE
