"""The explicit query/document token contract shared with JavaScript."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


@dataclass
class Batch:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    score_mask: np.ndarray
    truncated: list[bool]


class Preprocessor:
    def __init__(self, artifacts: Path):
        self.config = json.loads((artifacts / "runtime.json").read_text())
        self.tokenizer = Tokenizer.from_file(str(artifacts / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()

    def prepare(
        self, texts: list[str], kind: str, *, truncate: bool = False, document_length: int = 512
    ) -> Batch:
        if kind not in ("query", "document"):
            raise ValueError("kind must be query or document")
        if not texts or not all(isinstance(x, str) for x in texts):
            raise ValueError("Expected a nonempty list of strings")
        if (
            isinstance(document_length, bool)
            or not isinstance(document_length, int)
            or not (3 <= document_length <= self.config["max_document_length"])
        ):
            raise ValueError("document_length must be an integer in [3, 512]")
        limit = self.config["query_length"] if kind == "query" else document_length
        prefix = self.config["prefixes"][kind]
        rows, truncated = [], []
        for text in texts:
            ids = self.tokenizer.encode(prefix + text).ids
            too_long = len(ids) > limit
            if too_long:
                if not truncate:
                    raise ValueError(
                        f"{kind} has {len(ids)} tokens; limit is {limit}; "
                        "set truncate=True to explicitly allow truncation"
                    )
                ids = ids[: limit - 1] + [self.config["sep_token_id"]]
            rows.append(ids)
            truncated.append(too_long)
        width = limit if kind == "query" else max(map(len, rows))
        ids = np.full((len(rows), width), self.config["pad_token_id"], dtype=np.int64)
        attention = np.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = row
            attention[i, : len(row)] = 1
        score_mask = (
            np.ones_like(ids, dtype=bool)
            if kind == "query"
            else (attention.astype(bool) & ~np.isin(ids, self.config["skip_token_ids"]))
        )
        return Batch(ids, attention, score_mask, truncated)
