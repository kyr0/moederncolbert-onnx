import json
import os
import re

import numpy as np
import pytest

from moederncolbert_onnx.card import USAGE_MARKER
from moederncolbert_onnx.contracts import CONTRACTS
from moederncolbert_onnx.runtime import Encoder
from moederncolbert_onnx.source import LANGUAGES, MODELS, sha256


@pytest.mark.parametrize("slug", sorted(MODELS))
def test_folder_holds_exactly_the_upload_files(built, slug):
    folder, spec = built[slug], MODELS[slug]
    manifest = json.loads((folder / "manifest.json").read_text())
    expected = {
        "README.md", "LICENSE", "manifest.json", "parity-report.json", "runtime.json",
        "tokenizer.json", "tokenizer_config.json", *(f"model_{p}.onnx" for p in spec.precisions),
    }
    assert {p.name for p in folder.iterdir()} == expected
    assert list(manifest["models"]) == list(spec.precisions)
    for name, entry in manifest["files"].items():
        assert (folder / name).stat().st_size == entry["bytes"]
        assert sha256(folder / name) == entry["sha256"]
    for precision, model in manifest["models"].items():
        assert model["parity"] == CONTRACTS[precision]["parity"]
    assert (manifest["model_id"], manifest["revision"], manifest["repo_id"]) == (
        spec.model_id, spec.revision, spec.repo_id,
    )


@pytest.mark.parametrize("slug", sorted(MODELS))
def test_model_card_front_matter_and_numbers_come_from_the_report(built, slug):
    folder, spec = built[slug], MODELS[slug]
    card = (folder / "README.md").read_text()
    report = json.loads((folder / "parity-report.json").read_text())
    front = card.split("---\n")[1]
    assert "license: apache-2.0" in front and f"base_model: {spec.model_id}" in front
    assert all(f"- {code}\n" in front for code in LANGUAGES)
    assert f"`{spec.revision}`" in card
    for precision, result in report["precisions"].items():
        if precision == "fp32":
            continue
        cosine = f"{result['vs_fp32_retrieval_tokens']['mean_token_cosine']:.4f}"
        assert re.search(rf"^\| {precision} \| {cosine} \|", card, re.MULTILINE), precision
    for precision, reason in spec.excluded.items():
        assert f"{precision} is not shipped: {reason}." in card


@pytest.mark.parametrize("slug", sorted(MODELS))
def test_model_card_python_usage_runs_and_matches_the_encoder(built, slug, monkeypatch, capsys):
    folder = built[slug]
    card = (folder / "README.md").read_text()
    code = card.split(USAGE_MARKER, 1)[1].split("```", 1)[0]
    monkeypatch.chdir(folder.parent)
    namespace = {}
    exec(compile(code, "model-card-usage", "exec"), namespace)  # noqa: S102 - our own generated card
    assert "Zugspitze" in capsys.readouterr().out.split("'")[1]
    expected = Encoder(folder, "q8").rerank(
        "Wie hoch ist die Zugspitze?", ["Die Zugspitze ist 2962 Meter hoch.", "Paris is the capital of France."]
    )
    by_index = sorted(expected, key=lambda r: r["index"])
    np.testing.assert_allclose(namespace["scores"], [r["score"] for r in by_index], atol=1e-4)
    assert os.path.basename(str(folder)) in code
