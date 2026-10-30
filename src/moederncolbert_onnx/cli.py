"""`moedern-onnx` command line. Every command takes --model, and the ModelSpec derives all default paths,
so one model's files never land in the other model's upload folder.
VERIFIED: tests/test_models.py checks the derived paths and runs demo and reference."""

import argparse
import json
from pathlib import Path

from .source import MODELS, model_spec

COMMANDS = ("download", "export", "quantize", "parity", "build", "reference", "demo", "bench")


def parse(argv=None):
    parser = argparse.ArgumentParser(
        prog="moedern-onnx",
        description="Pinned MoEdernColBERT ONNX export, variants, parity checks and inference",
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--source", type=Path, help="default: upstream/<model>")
    parser.add_argument("--artifacts", type=Path, help="upload folder; default: output/hf/<repo name>")
    parser.add_argument("--fixture", type=Path, default=Path("fixtures/retrieval.json"))
    parser.add_argument("--reports", type=Path, help="default: reports/<model>")
    parser.add_argument(
        "--parity-fixture", type=Path, help="default: output/parity/<repo name>.json"
    )
    parser.add_argument("--out", type=Path, help="reference output; default: output/reference/<model>.json")
    parser.add_argument("--precision", default="q8")
    parser.add_argument("--query", default="Wie hoch ist die Zugspitze?")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args(argv)
    spec = model_spec(args.model)
    args.source = args.source or Path("upstream") / spec.slug
    args.artifacts = args.artifacts or Path("output/hf") / spec.repo_name
    args.reports = args.reports or Path("reports") / spec.slug
    args.parity_fixture = args.parity_fixture or Path("output/parity") / f"{spec.repo_name}.json"
    args.out = args.out or Path("output/reference") / f"{spec.slug}.json"
    return spec, args


def main(argv=None):
    spec, args = parse(argv)
    # Heavy imports (torch, ONNX) only after argument errors had their chance to fail fast.
    from .exporting import build_variants, export_fp32, write_json
    from .source import download
    from .validation import benchmark, reference_fixture, verify

    if args.command in ("download", "build"):
        download(spec, args.source)
        print(f"Verified pinned source: {args.source}")
    if args.command in ("export", "build"):
        export_fp32(spec, args.source, args.artifacts)
        print(f"FP32 exported: {args.artifacts / 'model_fp32.onnx'}")
    if args.command in ("quantize", "build"):
        manifest = build_variants(spec, args.artifacts)
        print(f"Variants built: {', '.join(manifest['models'])} in {args.artifacts}")
    if args.command in ("parity", "build"):
        report = verify(
            spec, args.source, args.artifacts, args.fixture, args.reports, args.parity_fixture
        )
        print(json.dumps({p: r["passed"] for p, r in report["precisions"].items()}))
    if args.command == "reference":
        write_json(args.out, reference_fixture(args.artifacts, args.fixture))
        print(f"Reference scores: {args.out}")
    if args.command == "bench":
        report = benchmark(args.artifacts, args.fixture, args.repeats)
        write_json(args.reports / "python-benchmark.json", report)
        print(json.dumps(report["results"], indent=2))
    if args.command == "demo":
        from .runtime import Encoder

        documents = json.loads(args.fixture.read_text())["documents"]
        encoder = Encoder(args.artifacts, args.precision)
        print(
            json.dumps(encoder.rerank(args.query, documents, top_k=5), indent=2, ensure_ascii=False)
        )


if __name__ == "__main__":
    main()
