import json
from typing import Dict, List


def genrecedit_load_json(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
