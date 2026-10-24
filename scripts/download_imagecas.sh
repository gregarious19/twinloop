#!/usr/bin/env bash
# usage: download_imagecas.sh <slug> [<slug> ...]   (runs inside the container)
# Downloads the Kaggle dataset(s) into /data/raw/imagecas. Plain .zip files are extracted in place.
# The official ImageCAS upload is a set of split archives (*.change2zip + *.z01..): Python cannot
# read those; join them on the host with scripts/join_split_zips.sh.
set -euo pipefail
DEST=/data/raw/imagecas
mkdir -p "$DEST"
for slug in "$@"; do
  echo "downloading $slug"
  kaggle datasets download -d "$slug" -p "$DEST" --unzip
done
while IFS= read -r -d '' z; do
  case "$z" in *"(1)"*) continue ;; esac
  d="$(dirname "$z")"
  if python -m zipfile -e "$z" "$d"; then rm -- "$z"; else echo "extract failed: $z"; exit 1; fi
done < <(find "$DEST" -name '*.zip' -print0)
if [ -n "$(find "$DEST" -name '*.change2zip' -print -quit)" ]; then
  echo "SPLIT ARCHIVES PRESENT: run scripts/join_split_zips.sh on the host"
fi
echo DOWNLOAD DONE
