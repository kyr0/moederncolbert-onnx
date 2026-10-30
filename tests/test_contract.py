import json
import shutil

import numpy as np
import pytest

from moederncolbert_onnx.runtime import Encoder, maxsim
from moederncolbert_onnx.tokenization import Preprocessor


def test_query_expansion_and_document_padding_are_different(built):
    prep = Preprocessor(built["1.5m"])
    query = prep.prepare(["cat"], "query")
    assert query.input_ids.shape == (1, 32)
    assert query.score_mask.all()
    assert query.attention_mask.sum() < 32
    assert query.input_ids[0, 1] == 30000
    docs = prep.prepare(["cat", "dog, bird, elephant!"], "document")
    assert not docs.score_mask[docs.attention_mask == 0].any()
    assert docs.score_mask.sum() < docs.attention_mask.sum()
    assert docs.input_ids[0, 1] == 30001


def test_truncation_is_explicit_and_preserves_separator(built):
    prep = Preprocessor(built["1.5m"])
    with pytest.raises(ValueError, match="limit"):
        prep.prepare(["the " * 600], "document")
    batch = prep.prepare(["the " * 600], "document", truncate=True)
    assert batch.truncated == [True]
    assert batch.input_ids.shape == (1, 512)
    assert batch.input_ids[0, -1] == 3
    for invalid in (0, 513, 5.5, True):
        with pytest.raises(ValueError):
            prep.prepare(["x"], "document", document_length=invalid)


def test_maxsim_negative_scores_and_validation():
    query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    doc = np.array([[-1.0, -2.0], [-3.0, -1.0]], dtype=np.float32)
    assert maxsim(query, doc) == -2
    with pytest.raises(ValueError):
        maxsim(query, np.empty((0, 2)))
    with pytest.raises(ValueError):
        maxsim(query, np.array([[np.nan, 0]]))


@pytest.mark.parametrize(
    ("slug", "precision"),
    [("1.5m", p) for p in ("fp32", "fp16", "q8", "int8")]
    + [("27m", p) for p in ("fp32", "fp16", "q8", "int8", "q4")],
)
def test_real_inference_batching_and_rank_ties(built, slug, precision):
    encoder = Encoder(built[slug], precision)
    texts = ["cat", "Die Zugspitze ist 2962 Meter hoch.", "Water freezes at zero degrees."]
    single = encoder.encode(texts, "document", batch_size=1)
    batch = encoder.encode(texts, "document", batch_size=3)
    for a, b in zip(single, batch, strict=True):
        # Dynamic activation quantization depends on the batch; every other variant must not.
        if precision != "int8":
            np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(b, axis=1), 1, atol=1e-3)
    assert [r["index"] for r in encoder.rerank("cat", ["cat", "cat"])] == [0, 1]
    assert encoder.encode([], "query") == []
    assert encoder.rerank("cat", [], top_k=1) == []
    with pytest.raises(ValueError):
        encoder.encode(texts, "query", batch_size=0)


def test_fingerprint_names_the_runtime_only_for_quality_precisions(built):
    manifest = json.loads((built["27m"] / "manifest.json").read_text())
    for precision, model in manifest["models"].items():
        fingerprint = Encoder(built["27m"], precision).fingerprint
        base = f"{manifest['revision']}:{model['sha256']}"
        if model["parity"] == "exact":
            assert fingerprint == base
        else:
            assert fingerprint.startswith(base + ":onnxruntime@") and fingerprint.endswith("/cpu")


def test_unshipped_precision_and_checksum_mismatch_fail_before_session(built, tmp_path):
    with pytest.raises(ValueError, match="precision must be one of"):
        Encoder(built["1.5m"], "q4")
    shutil.copyfile(built["1.5m"] / "manifest.json", tmp_path / "manifest.json")
    (tmp_path / "model_fp32.onnx").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        Encoder(tmp_path, "fp32")
