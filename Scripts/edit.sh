#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cls_model_path=""
# category="Video_Games"

CATEGORIES=(Cell_Phones_and_Accessories) # Software  "Video_Games" "Office_Products" "Cell_Phones_and_Accessories" "Musical_Instruments"
EDIT_POSTFIXES=(cold_test_augmented) #insert_similar  warm 
COV_LAMBDAS=(1000) # 3000 5000 6000 8000 10000 12000 # 1000 3000 5000
NUMBER_KNOWLEDGES=(10)
POS2LAYER=(0 1 2 3)
POS2LAYER_CONFIG="[0,1,2,3]"
for category in "${CATEGORIES[@]}"; do

  if [[ "$category" == "Video_Games" ]]; then
    max_rows=0.5
  elif [[ "$category" == "Office_Products" ]]; then
    max_rows=0.1
  elif [[ "$category" == "Cell_Phones_and_Accessories" ]]; then
    max_rows=0.1
  elif [[ "$category" == "Musical_Instruments" ]]; then
    max_rows=0.5
  elif [[ "$category" == "Industrial_and_Scientific" ]]; then
    max_rows=0.5
  elif [[ "$category" == "Software" ]]; then
    max_rows=0.5
  elif [[ "$category" == "Baby_Products" ]]; then
    max_rows=0.2
  else
    echo "Unknown category: $category"
    exit 1
  fi

  for edit_request_postfix in "${EDIT_POSTFIXES[@]}"; do
    for cov_lambda in "${COV_LAMBDAS[@]}"; do
      for number_knowledge in "${NUMBER_KNOWLEDGES[@]}"; do
        echo "===== postfix=${edit_request_postfix}  cov_lambda=${cov_lambda} ====="
        python edit_main.py \
          --category="${category}"\
          --pretrained_model_path="data/ckpt/TIGER_${category}/genrec_default_ori.pth"\
          --cov_lambda="${cov_lambda}" \
          --number_knowledge="${number_knowledge}"\
          --pos2layer "${POS2LAYER[@]}" \
          --covariance_data_file="data/Edit/${category}/edit_requests_COV.json" \
          --edit_requests_file="data/Edit/${category}/edit_requests_${edit_request_postfix}_${number_knowledge}.json" \
          --edit_name="edit_requests_${edit_request_postfix}"\
          --cache_dir="data/cache/"\
          --log_dir="outputs/logs/"\
          --tensorboard_log_dir="outputs/tensorboard/"\
          --max_rows="${max_rows}"

        python rec_main.py \
          --category="${category}"\
          --pretrained_model_path="data/ckpt/TIGER_${category}/genrec_default_ori.pth"\
          --deltaW_path="results/${category}/deltaW_edit_requests_${edit_request_postfix}_${cov_lambda}_${number_knowledge}.pt"\
          --cls_model_path="${cls_model_path}"\
          --cache_dir="data/cache/"\
          --log_dir="outputs/logs/"\
          --tensorboard_log_dir="outputs/tensorboard/"\
          --pos2layer="${POS2LAYER_CONFIG}"\
          --max_rows="${max_rows}"
      done
    done
  done
done
