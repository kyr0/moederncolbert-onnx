from pathlib import Path

import pytest

from moederncolbert_onnx.exporting import build_variants, export_fp32
from moederncolbert_onnx.source import MODELS
from moederncolbert_onnx.validation import verify

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures/retrieval.json"


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """Both models through the real pipeline (pinned source, export, variants, parity gates)."""
    folders = {}
    for slug, spec in MODELS.items():
        work = tmp_path_factory.mktemp(slug)
        folder = work / "hf" / spec.repo_name
        export_fp32(spec, ROOT / "upstream" / slug, folder)
        build_variants(spec, folder)
        verify(spec, ROOT / "upstream" / slug, folder, FIXTURE, work / "reports", work / "parity.json")
        folders[slug] = folder
    return folders
