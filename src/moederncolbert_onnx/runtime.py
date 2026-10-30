"""CPU ONNX runtime and pure MaxSim scoring."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .source import sha256
from .tokenization import Preprocessor


def maxsim(query: np.ndarray, document: np.ndarray) -> float:
    if query.ndim != 2 or document.ndim != 2 or query.shape[1] != document.shape[1]:
        raise ValueError("Expected token matrices with equal dimensions")
    if not len(query) or not len(document):
        raise ValueError("Token matrices must not be empty")
    if not np.isfinite(query).all() or not np.isfinite(document).all():
        raise ValueError("Token matrices must contain finite values")
    return float((query @ document.T).max(axis=1).sum(dtype=np.float64))


class Encoder:
    def __init__(self, artifacts: Path, precision: str = "q8", *, threads: int = 2):
        if precision not in ("fp32", "q8"):
            raise ValueError("precision must be fp32 or q8")
        manifest = json.loads((artifacts / "manifest.json").read_text())
        filename = manifest["models"][precision]["file"]
        for name in (filename, "runtime.json", "tokenizer.json", "tokenizer_config.json"):
            if sha256(artifacts / name) != manifest["files"][name]["sha256"]:
                raise ValueError(f"Artifact checksum mismatch: {name}")
        self.preprocessor = Preprocessor(artifacts)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(artifacts / filename), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def encode(
        self,
        texts: list[str],
        kind: str,
        *,
        batch_size: int = 4,
        truncate: bool = False,
        document_length: int = 512,
    ) -> list[np.ndarray]:
        if kind not in ("query", "document"):
            raise ValueError("kind must be query or document")
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise ValueError("texts must be a list of strings")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        output = []
        for start in range(0, len(texts), batch_size):
            batch = self.preprocessor.prepare(
                texts[start : start + batch_size],
                kind,
                truncate=truncate,
                document_length=document_length,
            )
            result = self.session.run(
                ["token_embeddings"],
                {
                    "input_ids": batch.input_ids,
                    "attention_mask": batch.attention_mask,
                },
            )[0]
            if not np.isfinite(result).all():
                raise ValueError("Non-finite encoder output")
            output.extend(result[i, mask].copy() for i, mask in enumerate(batch.score_mask))
        return output

    def rerank(
        self, query: str, documents: list[str], *, top_k: int | None = None, truncate: bool = False
    ) -> list[dict]:
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0
        ):
            raise ValueError("top_k must be a nonnegative integer")
        if not documents:
            return []
        q = self.encode([query], "query", truncate=truncate)[0]
        docs = self.encode(documents, "document", truncate=truncate)
        ranked = sorted(
            ({"index": i, "score": maxsim(q, d), "text": documents[i]} for i, d in enumerate(docs)),
            key=lambda item: (-item["score"], item["index"]),
        )
        return ranked if top_k is None else ranked[:top_k]
