from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MAX_ROWS = {
    "Video_Games": 0.5,
    "Office_Products": 0.1,
    "Cell_Phones_and_Accessories": 0.1,
    "Musical_Instruments": 0.5,
    "Industrial_and_Scientific": 0.5,
    "Software": 0.5,
    "Baby_Products": 0.2,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare GenRecEdit request JSON files.")
    parser.add_argument("--category", type=str, default="Cell_Phones_and_Accessories")
    parser.add_argument("--model_name", type=str, default="TIGER")
    parser.add_argument("--dataset_name", type=str, default="AmazonReviews2023")
    parser.add_argument("--cache_dir", type=str, default="data/cache/")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--max_rows", type=float, default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--number_per_item", type=int, default=10)
    parser.add_argument("--augmented_split", type=str, default="cold_test", choices=["cold_test", "warm_test", "test"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--write_cov", action="store_true", default=True)
    parser.add_argument("--no_write_cov", dest="write_cov", action="store_false")
    parser.add_argument("--write_augmented", action="store_true", default=True)
    parser.add_argument("--no_write_augmented", dest="write_augmented", action="store_false")
    parser.add_argument("--save_tokenized_augmented", action="store_true")
    return parser.parse_args()


def build_config(args):
    from accelerate import Accelerator

    from genrec.utils import get_config, init_device, init_seed

    max_rows = args.max_rows
    if max_rows is None:
        max_rows = DEFAULT_MAX_ROWS.get(args.category)
    if max_rows is None:
        raise ValueError(f"No default max_rows for category {args.category!r}; pass --max_rows explicitly.")

    pretrained_model_path = args.pretrained_model_path
    if pretrained_model_path is None:
        pretrained_model_path = f"data/ckpt/TIGER_{args.category}/genrec_default_ori.pth"

    config = get_config(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        config_file=None,
        config_dict={
            "category": args.category,
            "max_rows": max_rows,
            "cache_dir": args.cache_dir,
            "pretrained_model_path": pretrained_model_path,
            "log_dir": "outputs/logs/",
            "tensorboard_log_dir": "outputs/tensorboard/",
        },
    )
    config["device"], config["use_ddp"] = init_device()
    config["accelerator"] = Accelerator(
        log_with="tensorboard",
        project_dir=os.path.join(config["tensorboard_log_dir"], config["dataset"], config["model"]),
    )
    init_seed(config["rand_seed"], config["reproducibility"])
    return config


def build_edit_requests_json(tokenized_datasets, split: str) -> List[Dict]:
    import torch

    dataset = tokenized_datasets[split]
    edit_requests = []

    for i in range(len(dataset)):
        example = dataset[i]
        input_ids = example["input_ids"]
        labels = example["labels"]

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()

        edit_requests.append(
            {
                "history": input_ids,
                "target_sids": [str(x) for x in labels],
                "case_id": str(len(edit_requests)),
            }
        )

    return edit_requests


def load_sentence_embeddings(config: Dict, category: str) -> np.ndarray:
    import numpy as np

    sent_emb_path = Path(config["cache_dir"]) / config["dataset"] / category / "processed" / "sentence-t5-base.sent_emb"
    if not sent_emb_path.exists():
        raise FileNotFoundError(
            f"Sentence embedding file not found: {sent_emb_path}. "
            "Run the TIGER tokenizer once or copy the processed cache into data/cache/."
        )
    return np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, config["sent_emb_dim"])


def build_augmented_split(
    sent_embs: np.ndarray,
    split_datasets: Dict[str, Dataset],
    item2id: Dict[str, int],
    topk: int,
    max_aug_per_target: Optional[int],
    key_name: str,
    seed: Optional[int],
) -> Tuple[List[Dict[str, List[str]]], Dataset]:
    import numpy as np
    from datasets import Dataset

    rng = np.random.default_rng(seed)
    train_ds = split_datasets["train"]
    target_ds = split_datasets[key_name]

    train_items = set()
    for seq in train_ds["item_seq"]:
        train_items.update(seq)
    train_items.discard("[PAD]")

    train_emb_indices = []
    train_items_list = []
    for item in train_items:
        item_id = item2id.get(item)
        if item_id is None or item_id <= 0:
            continue
        emb_idx = item_id - 1
        if 0 <= emb_idx < sent_embs.shape[0]:
            train_emb_indices.append(emb_idx)
            train_items_list.append(item)

    train_emb_indices = np.array(train_emb_indices, dtype=np.int64)
    train_embs = sent_embs[train_emb_indices]
    train_embs_norm = train_embs / np.maximum(np.linalg.norm(train_embs, axis=1, keepdims=True), 1e-12)
    train_items_array = np.array(train_items_list, dtype=object)

    target_items = []
    seen = set()
    for seq in target_ds["item_seq"]:
        if not seq:
            continue
        target = seq[-1]
        if target not in seen:
            seen.add(target)
            target_items.append(target)

    def item_to_embidx(item: str) -> Optional[int]:
        item_id = item2id.get(item)
        if item_id is None or item_id <= 0:
            return None
        emb_idx = item_id - 1
        if 0 <= emb_idx < sent_embs.shape[0]:
            return emb_idx
        return None

    target2sims: Dict[str, List[str]] = {}
    mapping_list: List[Dict[str, List[str]]] = []
    for target in target_items:
        target_idx = item_to_embidx(target)
        if target_idx is None:
            target2sims[target] = []
            mapping_list.append({target: []})
            continue

        target_vec = sent_embs[target_idx]
        target_norm = np.linalg.norm(target_vec)
        if target_norm < 1e-12:
            target2sims[target] = []
            mapping_list.append({target: []})
            continue

        sims = train_embs_norm @ (target_vec / target_norm)
        k = min(topk, sims.shape[0])
        if k <= 0:
            target2sims[target] = []
            mapping_list.append({target: []})
            continue

        topk_idx = np.argpartition(-sims, kth=k - 1)[:k]
        topk_idx = topk_idx[np.argsort(-sims[topk_idx])]
        sim_items = train_items_array[topk_idx].tolist()
        target2sims[target] = sim_items
        mapping_list.append({target: sim_items})

    aug_users = []
    aug_item_seqs = []
    for user, seq in zip(train_ds["user"], train_ds["item_seq"]):
        for pos, item in enumerate(seq):
            if pos == 0:
                continue
            for target, sim_items in target2sims.items():
                if item not in sim_items:
                    continue
                aug_users.append(user)
                aug_item_seqs.append(seq[:pos] + [target])

    if max_aug_per_target is not None:
        grouped = {}
        for user, seq in zip(aug_users, aug_item_seqs):
            grouped.setdefault(seq[-1], []).append((user, seq))

        sampled = []
        for items in grouped.values():
            if len(items) > max_aug_per_target:
                selected = rng.choice(len(items), size=max_aug_per_target, replace=False)
                sampled.extend(items[int(i)] for i in selected)
            else:
                sampled.extend(items)
        aug_users = [user for user, _ in sampled]
        aug_item_seqs = [seq for _, seq in sampled]

    return mapping_list, Dataset.from_dict({"user": aug_users, "item_seq": aug_item_seqs})


def write_json(path: Path, payload: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(payload)} requests to {path}")


def main():
    args = parse_args()
    from genrec.utils import get_dataset, get_tokenizer

    output_dir = Path(args.output_dir or f"data/Edit/{args.category}")
    config = build_config(args)

    raw_dataset = get_dataset(args.dataset_name)(config)
    split_datasets = raw_dataset.split()
    tokenizer = get_tokenizer(args.model_name)(config, raw_dataset)

    if args.write_cov:
        tokenized_cov = tokenizer.tokenize({"train": split_datasets["train"]})
        cov_requests = build_edit_requests_json(tokenized_cov, "train")
        write_json(output_dir / "edit_requests_COV.json", cov_requests)

    if args.write_augmented:
        sent_embs = load_sentence_embeddings(config, args.category)
        _, augmented = build_augmented_split(
            sent_embs=sent_embs,
            split_datasets=split_datasets,
            item2id=raw_dataset.item2id,
            topk=args.topk,
            max_aug_per_target=args.number_per_item,
            key_name=args.augmented_split,
            seed=args.seed,
        )
        tokenized_augmented = tokenizer.tokenize({"cold_test_augmented": augmented})
        if args.save_tokenized_augmented:
            tokenized_path = output_dir / f"edit_requests_cold_test_augmented_{args.number_per_item}_train.json"
            tokenized_augmented["cold_test_augmented"].save_to_disk(str(tokenized_path))
            print(f"Saved tokenized augmented dataset to {tokenized_path}")

        augmented_requests = build_edit_requests_json(tokenized_augmented, "cold_test_augmented")
        write_json(output_dir / f"edit_requests_cold_test_augmented_{args.number_per_item}.json", augmented_requests)


if __name__ == "__main__":
    main()
