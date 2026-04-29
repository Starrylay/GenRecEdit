# GenRecEdit

GenRecEdit brings model editing to generative recommendation for cold-start scenarios. This repository contains a reproducible TIGER-based workflow for:

1. training a base generative recommender,
2. preparing cold-start edit requests,
3. solving and applying GenRecEdit weight updates,
4. evaluating the edited model.

The implementation follows the experimental pipeline used in:

```bibtex
@article{shen2026bringing,
  title={Bringing Model Editing to Generative Recommendation in Cold-Start Scenarios},
  author={Shen, Chenglei and Shi, Teng and Yu, Weijie and Zhang, Xiao and Xu, Jun},
  journal={arXiv preprint arXiv:2603.14259},
  year={2026}
}
```

## Repository Layout

```text
.
├── Scripts/
│   ├── rec_train.sh        # Train the base TIGER recommender
│   ├── prepare_data.sh     # Build GenRecEdit request JSON files
│   └── edit.sh             # Solve edits and evaluate the edited model
├── rec_main.py             # GenRec/TIGER training and evaluation entrypoint
├── edit_main.py            # GenRecEdit entrypoint
├── prepare_edit_data.py    # Reproducible data-preparation script
├── genrecedit/             # Editing algorithm
├── genrec/                 # Generative recommendation framework
├── util/                   # Hooking, statistics, and helper utilities
├── data/                   # Local datasets, caches, and checkpoints
├── results/                # Learned edit deltas
└── outputs/                # Logs and TensorBoard files
```

## Environment

Create a clean Python environment and install the project requirements:

```bash
cd GenRecEdit-open-source
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

The scripts use GPU `0` by default. Override it when needed:

```bash
CUDA_VISIBLE_DEVICES=1 bash Scripts/rec_train.sh Cell_Phones_and_Accessories
```

## Reproducible Workflow

Run the stages in this order.

### 1. Train The Base Recommender

Train TIGER for the target category:

```bash
bash Scripts/rec_train.sh Cell_Phones_and_Accessories
```

`rec_train.sh` writes checkpoints under:

```text
data/ckpt/TIGER_<category>/
```

The downstream data-preparation and editing scripts expect the base checkpoint to be named:

```text
data/ckpt/TIGER_<category>/genrec_default_ori.pth
```

After training, rename or copy the checkpoint produced by `rec_train.sh`:

```bash
category=Cell_Phones_and_Accessories
ckpt_dir="data/ckpt/TIGER_${category}"
latest_ckpt="$(ls -t "${ckpt_dir}"/genrec_default-*.pth | head -n 1)"
cp "${latest_ckpt}" "${ckpt_dir}/genrec_default_ori.pth"
```

This explicit rename is required because the next stages load `genrec_default_ori.pth`.

### 2. Prepare Edit Requests

Generate the covariance set and cold-start augmented edit requests:

```bash
bash Scripts/prepare_data.sh
```

The category list is configured in [Scripts/prepare_data.sh](Scripts/prepare_data.sh):

```bash
CATEGORIES=(Cell_Phones_and_Accessories)
```

For each category, the script writes:

```text
data/Edit/<category>/edit_requests_COV.json
data/Edit/<category>/edit_requests_cold_test_augmented_10.json
```

The preparation step requires the trained checkpoint from Step 1 and the TIGER processed cache under:

```text
data/cache/AmazonReviews2023/<category>/processed/
```

If the processed cache is missing, run the TIGER tokenizer/training path once or place the processed cache at the path above. The data loader may also download Amazon Reviews 2023 through HuggingFace `datasets` when raw data is not already cached.

### 3. Edit And Evaluate

Run the editing and evaluation script:

```bash
bash Scripts/edit.sh
```

The main knobs are defined near the top of [Scripts/edit.sh](Scripts/edit.sh):

```bash
CATEGORIES=(Cell_Phones_and_Accessories)
EDIT_POSTFIXES=(cold_test_augmented)
COV_LAMBDAS=(1000)
NUMBER_KNOWLEDGES=(10)
POS2LAYER=(0 1 2 3)
```

The edit stage loads:

```text
data/ckpt/TIGER_<category>/genrec_default_ori.pth
data/Edit/<category>/edit_requests_COV.json
data/Edit/<category>/edit_requests_cold_test_augmented_10.json
```

It saves the learned update to:

```text
results/<category>/deltaW_edit_requests_cold_test_augmented_<cov_lambda>_<number_knowledge>.pt
```

The evaluation stage then reloads the base checkpoint, applies the saved `deltaW`, and reports the edited-model metrics. Current evaluation output is intentionally compact and includes only:

```text
iid_ratio@K
ndcg@K
```

Here `ndcg@K` uses the prefix-level cold-start target matching used by GenRecEdit.

## Category Configuration

The default packaged category is:

```text
Cell_Phones_and_Accessories
```

To run another supported category, keep the category consistent across all stages:

1. pass it to `Scripts/rec_train.sh`,
2. update `CATEGORIES=(...)` in `Scripts/prepare_data.sh`,
3. update `CATEGORIES=(...)` in `Scripts/edit.sh`.

The current scripts define default `max_rows` values for:

```text
Video_Games
Cell_Phones_and_Accessories
Software
```

Add a category-specific `max_rows` branch in the scripts before running a new category.

## Outputs

Typical generated artifacts:

```text
data/ckpt/TIGER_<category>/genrec_default_ori.pth
data/Edit/<category>/edit_requests_COV.json
data/Edit/<category>/edit_requests_cold_test_augmented_10.json
results/<category>/deltaW_*.pt
outputs/logs/
outputs/tensorboard/
```

Large files such as checkpoints, processed caches, and edit-request JSON files should be handled with Git LFS or stored outside Git when publishing a lightweight code release.

## Troubleshooting

- `FileNotFoundError: genrec_default_ori.pth`: finish Step 1 and copy the trained checkpoint to `data/ckpt/TIGER_<category>/genrec_default_ori.pth`.
- Missing `sentence-t5-base.sent_emb`: build or copy the TIGER processed cache into `data/cache/AmazonReviews2023/<category>/processed/`.
- Argument mismatch between scripts: make sure `CATEGORIES`, `NUMBER_KNOWLEDGES`, `COV_LAMBDAS`, and `POS2LAYER` are aligned between `prepare_data.sh` and `edit.sh`.
- CUDA out of memory: reduce batch sizes in `genrec/default.yaml` or use a GPU with more memory.

## Citation

If this repository helps your research, please cite:

```bibtex
@article{shen2026bringing,
  title={Bringing Model Editing to Generative Recommendation in Cold-Start Scenarios},
  author={Shen, Chenglei and Shi, Teng and Yu, Weijie and Zhang, Xiao and Xu, Jun},
  journal={arXiv preprint arXiv:2603.14259},
  year={2026}
}
```
