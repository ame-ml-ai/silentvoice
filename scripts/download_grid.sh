#!/usr/bin/env bash
# Download the GRID corpus video + word alignments from Zenodo (CC BY 4.0).
# Audio (audio_25k.zip, 2.6 GB) is deliberately NOT downloaded: SilentVoice is visual-only.
#
#   bash scripts/download_grid.sh data/raw/grid          # all 33 talkers (~13.6 GB)
#   bash scripts/download_grid.sh data/raw/grid s1 s3 s5 # a subset
#
# Record: https://zenodo.org/records/3625687  (DOI 10.5281/zenodo.3625687)
set -euo pipefail

DEST="${1:?usage: download_grid.sh DEST [s1 s2 ...]}"
shift || true
BASE="https://zenodo.org/records/3625687/files"
if [ "$#" -gt 0 ]; then
  SPEAKERS=("$@")
else
  SPEAKERS=()
  for i in $(seq 1 34); do [ "$i" -ne 21 ] && SPEAKERS+=("s$i"); done  # s21 has no video
fi

mkdir -p "$DEST/_zips"
fetch() {
  local name="$1"
  if [ ! -s "$DEST/_zips/$name" ]; then
    echo "-> $name"
    curl -fL --retry 3 -o "$DEST/_zips/$name.part" "$BASE/$name?download=1"
    mv "$DEST/_zips/$name.part" "$DEST/_zips/$name"
  fi
}

fetch alignments.zip
unzip -qo "$DEST/_zips/alignments.zip" -d "$DEST"
for spk in "${SPEAKERS[@]}"; do
  fetch "$spk.zip"
  unzip -qo "$DEST/_zips/$spk.zip" -d "$DEST"
done

echo
echo "Layout check (expected: $DEST/<speaker>/*.mpg and $DEST/alignments/<speaker>/*.align):"
for spk in "${SPEAKERS[@]}"; do
  printf '  %-4s videos=%-5s alignments=%s\n' "$spk" \
    "$(find "$DEST/$spk" -name '*.mpg' 2>/dev/null | wc -l | tr -d ' ')" \
    "$(find "$DEST/alignments/$spk" -name '*.align' 2>/dev/null | wc -l | tr -d ' ')"
done
echo "If counts are 0, the archive layout differs: move files to match, then run prepare_dataset."
