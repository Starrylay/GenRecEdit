#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Select one category per run.
# Usage:
#   bash Scripts/rec_train_phone.sh Software
#   CATEGORY=Video_Games bash Scripts/rec_train_phone.sh

category="${1:-${CATEGORY:-Video_Games}}"  #Video_Games  Software  Cell_Phones_and_Accessories

max_rows_for_category() {
  local category="$1"
  case "${category}" in
    Video_Games) echo "0.5" ;;
    Cell_Phones_and_Accessories) echo "0.1" ;;
    Software) echo "0.5" ;;
    *)
      echo "Unknown category: ${category}" >&2
      return 1
      ;;
  esac
}

max_rows="$(max_rows_for_category "${category}")"
echo "===== Training TIGER category=${category} max_rows=${max_rows} ====="

python main.py \
  --model=TIGER \
  --category="${category}" \
  --max_rows="${max_rows}"
