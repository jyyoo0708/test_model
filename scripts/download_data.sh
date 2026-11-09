#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. Research/education only; do not redistribute images.
mkdir -p data/raw/crowdhuman
base='https://huggingface.co/datasets/Funtion/CrowdHuman/resolve/636b0dc73047458406c762aeb445cd75cee85b51'
for name in CrowdHuman_val.zip annotation_val.odgt; do
    if [ ! -f "data/raw/crowdhuman/$name" ]; then
        curl -fL --retry 3 --connect-timeout 30 "$base/$name" -o "data/raw/crowdhuman/$name.part"
        mv "data/raw/crowdhuman/$name.part" "data/raw/crowdhuman/$name"
    fi
done
sha256sum --check data/raw/crowdhuman/SHA256SUMS
