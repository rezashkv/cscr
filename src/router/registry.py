# src/router/registry.py
import json
from pathlib import Path

_REGISTRY_PATH = Path(__file__).parent.parent.parent / 'experts' / 'registry.json'
with open(_REGISTRY_PATH, 'r') as f:
    REGISTRY = json.load(f)