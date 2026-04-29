# GenRecEdit

GenRecEdit is a model-editing workflow for generative recommendation based on the TIGER/GenRec code path.

This repository was packaged from the runnable command in `Scripts/edit.sh`. The script first computes a `deltaW`
edit with `edit_position_诸多验证.py`, then evaluates the edited model through `main.py`.

## Repository Layout

```text
.
├── Scripts/edit.sh
├── edit_position_诸多验证.py
├── main.py
├── genrecedit/
├── genrec/
├── util/
├── data/
│   ├── Edit/Cell_Phones_and_Accessories/
│   ├── ckpt/TIGER_Cell_Phones_and_Accessories/
│   └── cache/AmazonReviews2023/Cell_Phones_and_Accessories/processed/
├── results/
└── outputs/
```

## Included Data

The `data/` directory contains the files referenced by the packaged `Scripts/edit.sh` command:

- `data/Edit/Cell_Phones_and_Accessories/edit_requests_COV.json`
- `data/Edit/Cell_Phones_and_Accessories/edit_requests_cold_test_augmented_10.json`
- `data/ckpt/TIGER_Cell_Phones_and_Accessories/genrec_default_ori.pth`
- `data/cache/AmazonReviews2023/Cell_Phones_and_Accessories/processed/`

The full HuggingFace dataset download cache is intentionally not copied into this repository. If it is absent, the
dataset loader may download Amazon Reviews 2023 files through `datasets.load_dataset` on first run.

## Run

```bash
bash Scripts/edit.sh
```

By default, the script uses GPU `0` unless `CUDA_VISIBLE_DEVICES` is already set:

```bash
CUDA_VISIBLE_DEVICES=1 bash Scripts/edit.sh
```

Generated files are written under:

- `results/`
- `outputs/logs/`
- `outputs/tensorboard/`

## Prepare Data

The data-generation notebooks are under `notebooks/`, and their runnable open-source version is:

```bash
bash Scripts/prepare_data.sh
```

See [docs/data_preparation.md](docs/data_preparation.md) for the full data flow.

## Notes for Publishing

The checkpoint and edit-request JSON files are large. Use Git LFS for files matched by `.gitattributes` before pushing
to a public Git host.
