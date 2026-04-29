# Data Preparation

The packaged experiment consumes two GenRecEdit request files:

- `data/Edit/Cell_Phones_and_Accessories/edit_requests_COV.json`
- `data/Edit/Cell_Phones_and_Accessories/edit_requests_cold_test_augmented_10.json`

The original exploratory notebooks are preserved under `notebooks/`:

- `notebooks/数据划分_covdata.ipynb`
- `notebooks/数据划分_train.ipynb`

For reproducible use in the open-source repository, the notebook logic has been consolidated into:

- `scripts/prepare_edit_data.py`
- `Scripts/prepare_data.sh`

## What The Files Mean

`edit_requests_COV.json` is built from the tokenized training split and is used by GenRecEdit to compute covariance
statistics.

`edit_requests_cold_test_augmented_10.json` is built by:

1. Loading the Amazon Reviews 2023 split for the selected category.
2. Finding cold-test target items.
3. Loading item sentence embeddings from `data/cache/.../processed/sentence-t5-base.sent_emb`.
4. Finding similar training items by cosine similarity.
5. Replacing similar training-item positions with the cold target to create augmented edit examples.
6. Tokenizing the augmented examples with the TIGER tokenizer.
7. Writing GenRecEdit request JSON entries with:
   - `history`: tokenized input sequence
   - `target_sids`: semantic ID target tokens
   - `case_id`: string index

## Run

```bash
bash Scripts/prepare_data.sh
```

Equivalent direct command:

```bash
python scripts/prepare_edit_data.py \
  --category Cell_Phones_and_Accessories \
  --number_per_item 10 \
  --topk 10 \
  --cache_dir data/cache/ \
  --output_dir data/Edit/Cell_Phones_and_Accessories
```

## Required Inputs

The script expects the processed TIGER cache to exist at:

```text
data/cache/AmazonReviews2023/Cell_Phones_and_Accessories/processed/
```

In particular, `sentence-t5-base.sent_emb` is required for similarity-based augmentation. If this file is missing,
run the TIGER tokenizer once or copy the processed cache into `data/cache/`.

The dataset loader may also use HuggingFace `datasets.load_dataset` for Amazon Reviews 2023 if raw dataset cache is not
already available.

## Relationship To Notebooks

The notebooks are kept as research records. Prefer `scripts/prepare_edit_data.py` for repeatable generation because it:

- uses repository-relative paths,
- exposes parameters as CLI flags,
- writes directly into `data/Edit/<category>/`,
- avoids hard-coded local paths from the original project workspace.
