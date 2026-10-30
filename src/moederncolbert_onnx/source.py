"""Pinned upstream download. Only this module accesses the model host."""

import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL_ID = "VAGOsolutions/SauerkrautLM-MoEdernColBERT-1.5M"
REVISION = "9d360b8b88ae11bea5784bf9012ff509fdd5c2da"
FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
    "1_Dense/config.json",
    "1_Dense/model.safetensors",
    "modeling_modernbert_hash_moe_v10.py",
    "moe_triton.py",
    "LICENSE",
    "README.md",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "source-manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if existing.get("model_id") != MODEL_ID or existing.get("revision") != REVISION:
        existing = {}

    def fetch(name: str) -> tuple[str, dict]:
        target = destination / name
        expected = existing.get("files", {}).get(name, {}).get("sha256")
        if not (target.exists() and expected and sha256(target) == expected):
            url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(target)
        return name, {"bytes": target.stat().st_size, "sha256": sha256(target)}

    with ThreadPoolExecutor(max_workers=4) as executor:
        entries = dict(executor.map(fetch, FILES))
    manifest = {"model_id": MODEL_ID, "revision": REVISION, "files": entries}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_source(path: Path) -> dict:
    manifest = json.loads((path / "source-manifest.json").read_text())
    if manifest["model_id"] != MODEL_ID or manifest["revision"] != REVISION:
        raise ValueError("This exporter supports only the pinned 1.5M revision")
    for name in FILES:
        if sha256(path / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Upstream checksum mismatch: {name}")
    return manifest
