import ast
import json
from dataclasses import dataclass, field
from typing import List


def _normalize_pos2layer(pos2layer):
    if pos2layer is None:
        return [0, 1, 2, 3]
    if isinstance(pos2layer, str):
        pos2layer = ast.literal_eval(pos2layer)
    if isinstance(pos2layer, tuple):
        pos2layer = list(pos2layer)
    if not isinstance(pos2layer, list) or not pos2layer:
        raise ValueError("GenRecEdit pos2layer must be a non-empty list of decoder layer indices.")
    return [int(layer_idx) for layer_idx in pos2layer]


@dataclass
class GenRecEditHyperParams:
    layers: List[int] = field(default_factory=lambda: [14, 15, 16, 17, 18])

    model_name: str = "UnKnown"
    edit_name: str = "UnKnown"

    decoder_module_tmp: str = "decoder.block.{}.layer.2.DenseReluDense.wo"
    pos2layer: List[int] = field(default_factory=lambda: [0, 1, 2, 3])

    v_loss_layer: int = -1
    v_lr: float = 0.5
    v_num_grad_steps: int = 30
    v_weight_decay: float = 0.2
    clamp_norm_factor: float = 1
    kl_factor: float = 0.0625
    cov_lambda: int = 10000
    number_knowledge: int = 5

    fact_token: str = "last"

    mom2_n_samples: int = 400000
    mom2_dtype: str = "float32"
    mom2_eps: float = 1e-2
    mom2_update_weight: float = 1e-5
    covariance_cache_dir: str = "cache/covariance"
    covariance_data_file: str = ""

    pseudo_num: int = 10
    z_vector_max: int = 8000
    stats_batch_size: int = 1280

    use_prob_threshold: bool = True
    prob_threshold: float = 0.3

    def __post_init__(self):
        self.pos2layer = _normalize_pos2layer(self.pos2layer)

    def genrecedit_save(self, save_path: str):
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)
