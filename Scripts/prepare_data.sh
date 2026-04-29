#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CATEGORIES=(Cell_Phones_and_Accessories) # Software  "Video_Games" "Office_Products" "Cell_Phones_and_Accessories" "Musical_Instruments"

for category in "${CATEGORIES[@]}"; do
  python prepare_edit_data.py \
    --category="${category}" \
    --number_per_item=10 \
    --topk=10 \
    --cache_dir="data/cache/" \
    --output_dir="data/Edit/${category}"
done
