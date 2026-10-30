import json
import shutil
from pathlib import Path

import pytest

from moederncolbert_onnx.cli import main, parse
from moederncolbert_onnx.portable import PortableEncoder, check_architecture
from moederncolbert_onnx.source import MODELS, model_spec, verify_source

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("slug", sorted(MODELS))
def test_pinned_sources_verify_and_reject_modified_bytes(slug, tmp_path):
    spec = MODELS[slug]
    verify_source(spec, ROOT / "upstream" / slug)
    copy = tmp_path / slug
    shutil.copytree(ROOT / "upstream" / slug, copy)
    config = copy / "config.json"
    config.write_bytes(config.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum mismatch: config.json"):
        verify_source(spec, copy)
    config.unlink()
    with pytest.raises(ValueError, match="Missing upstream file config.json"):
        verify_source(spec, copy)


def test_both_specs_ship_fp32_first_and_explain_exclusions():
    for spec in MODELS.values():
        assert spec.precisions[0] == "fp32"
        assert not set(spec.excluded) & set(spec.precisions)
    assert "q4" in MODELS["1.5m"].excluded
    with pytest.raises(ValueError, match="Unknown model"):
        model_spec("115m")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"layer_types": ["attn_moe", "conv"]}, "layer_types"),
        ({"conv_dilations": [1, 2]}, "convolution"),
        ({"num_shared_experts": 2}, "shared experts"),
        ({"max_position_embeddings": 8192}, "positions beyond 512"),
        ({"num_key_value_heads": 3}, "head layout"),
    ],
)
def test_architecture_guard_rejects_unimplemented_layers(change, message):
    config = json.loads((ROOT / "upstream/27m/config.json").read_text())
    check_architecture(config)
    with pytest.raises(ValueError, match=message):
        check_architecture(config | change)


@pytest.mark.parametrize("slug", sorted(MODELS))
def test_portable_encoder_reads_dimensions_from_config(slug):
    config = json.loads((ROOT / "upstream" / slug / "config.json").read_text())
    encoder = PortableEncoder(ROOT / "upstream" / slug)
    assert len(encoder.blocks) == config["num_hidden_layers"]
    assert encoder.embedding_norm.normalized_shape == (config["hidden_size"],)
    assert encoder.projection.out_features == 128


def test_cli_requires_a_model_and_derives_paths(capsys):
    with pytest.raises(SystemExit):
        parse(["export"])
    assert "--model" in capsys.readouterr().err
    spec, args = parse(["parity", "--model", "27m"])
    assert args.artifacts == Path("output/hf/SauerkrautLM-MoEdernColBERT-27M-ONNX")
    assert args.parity_fixture == Path("output/parity/SauerkrautLM-MoEdernColBERT-27M-ONNX.json")
    assert (args.source, args.reports) == (Path("upstream/27m"), Path("reports/27m"))
    assert spec.repo_id == "kyr0/SauerkrautLM-MoEdernColBERT-27M-ONNX"


def test_cli_demo_and_reference_run_on_a_built_folder(built, tmp_path, capsys):
    folder = str(built["27m"])
    main(["demo", "--model", "27m", "--artifacts", folder, "--precision", "q4"])
    ranked = json.loads(capsys.readouterr().out)
    assert ranked[0]["text"].startswith("Die Zugspitze") and len(ranked) == 5
    out = tmp_path / "reference.json"
    main(["reference", "--model", "27m", "--artifacts", folder, "--out", str(out)])
    reference = json.loads(out.read_text())
    assert set(reference["precisions"]) == set(MODELS["27m"].precisions)
    for precision in reference["precisions"].values():
        scores = precision["scores"]
        assert len(scores) == len(reference["queries"]) and len(scores[0]) == len(reference["documents"])
    # Every picked query's labeled passage is among the documents and ranks first in FP32.
    for row, relevant in zip(reference["precisions"]["fp32"]["scores"], reference["relevant"], strict=True):
        assert row.index(max(row)) in relevant
