"""Pinned upstream checkpoints. Only this module accesses the model host.

VERIFIED: the pinned SHA-256 of every LFS file equals Hugging Face's x-linked-etag at the pinned revision.
"""

import hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# VERIFIED: both upstream cards list these languages (README front matter at the pinned revisions).
LANGUAGES = ("de", "en", "es", "nl", "fr", "it", "pt")
# Files shared byte-for-byte by both checkpoints: tokenizer, custom modeling code, license.
_SHARED = {
    "config_sentence_transformers.json": "007399a787035b4350ed00166bc77380a16fbf5b161153f9452e86b0cb3932a7",
    "modules.json": "02e279a7d7019ebac4183eb61ff776bb36688b0b01202849423bbc58850e99c5",
    "sentence_bert_config.json": "58ffae3751d252be0313f661207de131aa09f3dbaf80df1a95b7dafc5bf45e5f",
    "tokenizer.json": "7357662aa3be57e8fd953365a37f892dab4ee702b817e1efcef9f193a388aff0",
    "tokenizer_config.json": "fc250460005ed0c1d47fa58829a31ccfd57a5abacf89029c27271869a588dc18",
    "modeling_modernbert_hash_moe_v10.py": "528afffbd33210845ed521b12af5569f9b6d0b0eb3f176beebb3efd83cee6668",
    "moe_triton.py": "1ad1c291ec1d7ea6b35d02d24ffafba7a12fb916395c522864ab3616c62f339f",
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
}


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    model_id: str
    revision: str
    # Hugging Face repository that receives the converted folder.
    repo_id: str
    files: dict
    # Shipped ONNX variants, in build order; fp32 always comes first.
    precisions: tuple
    # Variants deliberately not shipped, with the measured reason (shown in the model card).
    excluded: dict = field(default_factory=dict)

    @property
    def repo_name(self) -> str:
        return self.repo_id.split("/")[1]


MODELS = {
    "1.5m": ModelSpec(
        slug="1.5m",
        model_id="VAGOsolutions/SauerkrautLM-MoEdernColBERT-1.5M",
        revision="9d360b8b88ae11bea5784bf9012ff509fdd5c2da",
        repo_id="kyr0/SauerkrautLM-MoEdernColBERT-1.5M-ONNX",
        files={
            **_SHARED,
            "config.json": "54f7e0dc02d5814b3bd9b2f2f7f1f2cbd732aef7c4de7f334eaf2daa43e75e6e",
            "model.safetensors": "3cd03bd9a9671a05f4c34dc3b3af30d0eb802f89d7f96923873dab06bdff8cfb",
            "1_Dense/config.json": "0b0dfe7f9ccd774f5e90f84256060b4bea73f371caf81967ee9c03c46b71fad9",
            "1_Dense/model.safetensors": "ef5c34a87882b098f044dfb30892e485ffe5cffe95d5ad78563d6f656666d6a9",
            "README.md": "5b7d2ba6ee057c9d0c8b81c0c76a5fe9ec6c2d09820119b7c82498406b522c2e",
        },
        precisions=("fp32", "fp16", "q8", "int8"),
        excluded={
            "q4": "its mean token cosine against FP32 measured 0.954 to 0.961 across block sizes "
            "16 and 32, below the 0.98 gate"
        },
    ),
    "27m": ModelSpec(
        slug="27m",
        model_id="VAGOsolutions/SauerkrautLM-MoEdernColBERT-27M",
        revision="d2b7b1feaaac1bf87510ca9525208c3daeabcc5a",
        repo_id="kyr0/SauerkrautLM-MoEdernColBERT-27M-ONNX",
        files={
            **_SHARED,
            "config.json": "4a4a1b8ab25f4ea0a8b111f48bbcb88f3005771ac56fac50fe8e569ffd3e06e1",
            "model.safetensors": "a258ed25e26ed92c449381af9e0da64f90df746fb1989268326a3656ccb60a59",
            "1_Dense/config.json": "934dadda2f5abfa68d6584585472935e35b06cd30fc14dd648802f437aac6f95",
            "1_Dense/model.safetensors": "e380c324221a4dfb1f0bebe35de2b741774965c9283370cc9c3e68af710e72eb",
            "README.md": "a107040dd0f4987cf741b3391bda58dd97a2296b0e3dd3b9dee7fa11fb02ab71",
        },
        precisions=("fp32", "fp16", "q8", "int8", "q4"),
    ),
}


def model_spec(slug: str) -> ModelSpec:
    if slug not in MODELS:
        raise ValueError(f"Unknown model {slug!r}; choose one of {', '.join(MODELS)}")
    return MODELS[slug]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(spec: ModelSpec, destination: Path) -> None:
    """Fetch missing or mismatching files of the pinned revision; never trust unchecked bytes."""

    def fetch(name: str) -> None:
        target = destination / name
        if target.exists() and sha256(target) == spec.files[name]:
            return
        url = f"https://huggingface.co/{spec.model_id}/resolve/{spec.revision}/{name}"
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != spec.files[name]:
            raise ValueError(f"Downloaded {name} does not match its pinned SHA-256")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(target)

    destination.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(fetch, spec.files))


def verify_source(spec: ModelSpec, path: Path) -> None:
    for name, expected in spec.files.items():
        if not (path / name).exists():
            raise ValueError(f"Missing upstream file {name}; run `moedern-onnx download`")
        if sha256(path / name) != expected:
            raise ValueError(f"Upstream checksum mismatch: {name}")
