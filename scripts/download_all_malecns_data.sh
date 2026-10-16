#!/usr/bin/env bash
# Explicit opt-in download of official Janelia MaleCNS v1.0 data.
# The full connection table is ~1.1GB. Never download at app startup.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
folder="data/malecns_full"
mkdir -p "$folder"
free_kb="$(df -Pk "$folder" | awk 'END {print $4}')"
if [[ "$free_kb" -lt 4000000 ]]; then
  echo "❌ At least 4 GB free disk space is needed for safe offline training." >&2
  exit 1
fi
base="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
for filename in \
  "body-annotations-male-cns-v1.0-minconf-0.5.feather" \
  "connectome-weights-male-cns-v1.0-minconf-0.5.feather"; do
  target="$folder/$filename"
  if [[ -s "$target" ]]; then
    echo "Already present: $target (training validates the file)"
    continue
  fi
  echo "Downloading official MaleCNS file: $filename"
  curl --fail --location --retry 3 --continue-at - --output "$target" "$base/$filename"
done
echo "✅ Official full-graph data ready for offline validation."
