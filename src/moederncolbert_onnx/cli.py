import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Pinned MoEdernColBERT export, Q8, parity and inference"
    )
    parser.add_argument(
        "command",
        choices=["download", "export", "quantize", "verify", "build", "demo", "benchmark"],
    )
    parser.add_argument("--source", type=Path, default=Path("upstream"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--fixture", type=Path, default=Path("fixtures/retrieval.json"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--q8-mode", choices=["weight-only", "dynamic"], default="weight-only")
    parser.add_argument("--precision", choices=["fp32", "q8"], default="q8")
    parser.add_argument("--query", default="Wie hoch ist die Zugspitze?")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    from .exporting import export_fp32, quantize_q8, write_json
    from .source import download
    from .validation import benchmark, verify

    if args.command in ("download", "build"):
        download(args.source)
        print(f"Verified pinned source: {args.source}")
    if args.command in ("export", "build"):
        export_fp32(args.source, args.artifacts)
        print(f"FP32 exported: {args.artifacts / 'model_fp32.onnx'}")
    if args.command in ("quantize", "build"):
        quantize_q8(args.artifacts, args.q8_mode)
        print(f"Q8 exported ({args.q8_mode}): {args.artifacts / 'model_q8.onnx'}")
    if args.command in ("verify", "build"):
        report = verify(args.source, args.artifacts, args.fixture, args.reports)
        print(json.dumps(report, indent=2))
    if args.command == "benchmark":
        report = benchmark(args.artifacts, args.fixture, args.repeats)
        write_json(args.reports / "python-benchmark.json", report)
        print(json.dumps(report, indent=2))
    if args.command == "demo":
        from .runtime import Encoder

        documents = json.loads(args.fixture.read_text())["documents"]
        encoder = Encoder(args.artifacts, args.precision)
        print(
            json.dumps(encoder.rerank(args.query, documents, top_k=5), indent=2, ensure_ascii=False)
        )


if __name__ == "__main__":
    main()
