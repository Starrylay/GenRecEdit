import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch


class GenRecEditCovarianceCache:
    def __init__(self, cache_dir: str = "cache/covariance"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def genrecedit_make_cache_key(
        self,
        model_name: str,
        module_name: str,
        cov_data_file: str,
        sample_size: int,
        dtype: str,
        position: int,
    ) -> str:
        try:
            with open(cov_data_file, "r", encoding="utf-8") as f:
                file_content = f.read()
            file_hash = hashlib.md5(file_content.encode()).hexdigest()
        except Exception:
            file_hash = "no_hash"

        cache_components = [
            model_name.replace("/", "_").replace("\\", "_"),
            module_name.replace(".", "_"),
            file_hash[:12],
            str(sample_size),
            dtype,
            str(position),
        ]
        return "_".join(cache_components)

    def _genrecedit_cache_matrix_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"cov_{cache_key}.pkl"

    def _genrecedit_cache_meta_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"meta_{cache_key}.json"

    def genrecedit_save_covariance(
        self,
        covariance_matrix: torch.Tensor,
        cache_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        cache_filepath = self._genrecedit_cache_matrix_path(cache_key)
        metadata_filepath = self._genrecedit_cache_meta_path(cache_key)

        try:
            with open(cache_filepath, "wb") as f:
                pickle.dump(covariance_matrix.cpu(), f)

            if metadata is None:
                metadata = {}

            metadata.update(
                {
                    "cache_key": cache_key,
                    "matrix_shape": list(covariance_matrix.shape),
                    "matrix_dtype": str(covariance_matrix.dtype),
                    "created_at": str(pd.Timestamp.now()) if "pandas" in sys.modules else "unknown",
                    "cache_version": "1.0",
                }
            )

            with open(metadata_filepath, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        except Exception:
            for filepath in (cache_filepath, metadata_filepath):
                if filepath.exists():
                    filepath.unlink()

    def genrecedit_load_covariance(self, cache_key: str) -> Optional[torch.Tensor]:
        cache_filepath = self._genrecedit_cache_matrix_path(cache_key)
        metadata_filepath = self._genrecedit_cache_meta_path(cache_key)

        if not cache_filepath.exists():
            return None

        try:
            if metadata_filepath.exists():
                with open(metadata_filepath, "r", encoding="utf-8") as f:
                    _ = json.load(f)

            with open(cache_filepath, "rb") as f:
                covariance_matrix = pickle.load(f)
            return covariance_matrix

        except Exception:
            for filepath in (cache_filepath, metadata_filepath):
                if filepath.exists():
                    try:
                        filepath.unlink()
                    except Exception:
                        pass
            return None
