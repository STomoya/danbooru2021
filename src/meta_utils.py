
from __future__ import annotations
import json

def load_as_dict(path) -> list[dict]:
    with open(path, 'r') as jsonl:
        lines = jsonl.read().strip().split('\n')
    return [json.loads(line) for line in lines]

def load_filelist(path) -> list[str]:
    with open(path, 'r') as fin:
        filelist = fin.read().strip().split('\n')
    return filelist
