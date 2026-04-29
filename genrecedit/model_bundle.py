import os

import torch
from accelerate import Accelerator

from util import nethook
from genrec.utils import (
    get_config,
    init_seed,
    init_device,
    get_dataset,
    get_tokenizer,
    get_model,
)


class GenRecEditModelBundle:
    def __init__(self, args):
        model_name = args.model_name
        pretrained_model_path = args.pretrained_model_path

        config_overrides = {}
        for key in ("cache_dir", "log_dir", "tensorboard_log_dir", "ckpt_dir", "pos2layer"):
            value = getattr(args, key, None)
            if value is not None:
                config_overrides[key] = value

        config = get_config(
            model_name=model_name,
            dataset_name="AmazonReviews2023",
            config_file=None,
            config_dict=config_overrides or None,
        )
        config["category"] = args.category
        config["max_rows"] = args.max_rows

        config["device"], config["use_ddp"] = init_device()
        project_dir = os.path.join(config["tensorboard_log_dir"], config["dataset"], config["model"])
        accelerator = Accelerator(log_with="tensorboard", project_dir=project_dir)
        config["accelerator"] = accelerator

        init_seed(config["rand_seed"], config["reproducibility"])

        raw_dataset = get_dataset("AmazonReviews2023")(config)
        tokenizer = get_tokenizer(model_name)(config, raw_dataset)

        with accelerator.main_process_first():
            model = get_model(model_name)(config, raw_dataset, tokenizer)

        model.load_state_dict(torch.load(pretrained_model_path))
        nethook.set_requires_grad(False, model)
        model.eval().cuda()

        self.tokenizer = tokenizer
        self.model = model
        self.num_decoder_layers = len(list(model.decoder.block)) if hasattr(model, "decoder") else 0

    def __repr__(self):
        return (
            f"GenRecEditModelBundle(model: {type(self.model).__name__} [T5-style, "
            f"{self.num_decoder_layers} decoder layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )
