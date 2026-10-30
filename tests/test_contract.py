import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from moederncolbert_onnx.runtime import Encoder, maxsim
from moederncolbert_onnx.tokenization import Preprocessor

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def test_query_expansion_and_document_padding_are_different():
    prep = Preprocessor(ARTIFACTS)
    query = prep.prepare(["cat"], "query")
    assert query.input_ids.shape == (1, 32)
    assert query.score_mask.all()
    assert query.attention_mask.sum() < 32
    assert query.input_ids[0, 1] == 30000
    docs = prep.prepare(["cat", "dog, bird, elephant!"], "document")
    assert not docs.score_mask[docs.attention_mask == 0].any()
    assert docs.score_mask.sum() < docs.attention_mask.sum()
    assert docs.input_ids[0, 1] == 30001


def test_truncation_is_explicit_and_preserves_separator():
    prep = Preprocessor(ARTIFACTS)
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


@pytest.mark.parametrize("precision", ["fp32", "q8"])
def test_real_inference_batching_and_rank_ties(precision):
    encoder = Encoder(ARTIFACTS, precision)
    texts = ["cat", "Die Zugspitze ist 2962 Meter hoch.", "Water freezes at zero degrees."]
    single = encoder.encode(texts, "document", batch_size=1)
    batch = encoder.encode(texts, "document", batch_size=3)
    for a, b in zip(single, batch, strict=True):
        # Dynamic activation quantization is batch-dependent; default weight-only is not.
        mode = (
            json.loads((ARTIFACTS / "manifest.json").read_text())["models"][precision]
            .get("quantization", {})
            .get("mode")
        )
        if mode != "dynamic":
            np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(b, axis=1), 1, atol=1e-5)
    assert [r["index"] for r in encoder.rerank("cat", ["cat", "cat"])] == [0, 1]
    assert encoder.encode([], "query") == []
    assert encoder.rerank("cat", [], top_k=1) == []
    with pytest.raises(ValueError):
        encoder.encode(texts, "query", batch_size=0)


def test_checksum_mismatch_fails_before_session_creation(tmp_path):
    shutil.copyfile(ARTIFACTS / "manifest.json", tmp_path / "manifest.json")
    (tmp_path / "model_fp32.onnx").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        Encoder(tmp_path, "fp32")
