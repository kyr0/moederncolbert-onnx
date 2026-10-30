.DEFAULT_GOAL := help
.PHONY: help setup download export quantize verify test browser-install browser-test demo demo-node benchmark all

help:
	@echo "make setup         Install locked Python and JavaScript dependencies"
	@echo "make all           Rebuild models, verify Python/Node, run tests, build browser demo"
	@echo "make demo          Start local interactive browser demo"
	@echo "make demo-node     Run the Node.js Q8 example"
	@echo "make browser-test  Install Chromium and verify WASM + browser UI"
	@echo "make benchmark     Measure FP32/Q8 performance on this machine"

setup:
	uv sync --frozen --python 3.11
	ONNXRUNTIME_NODE_INSTALL=skip bun install --frozen-lockfile
	bun run build

download:
	uv run --frozen moedern-onnx download

export:
	uv run --frozen moedern-onnx export

quantize:
	uv run --frozen moedern-onnx quantize

verify:
	uv run --frozen moedern-onnx verify
	bun run build
	bun run test:node

test:
	uv run --frozen pytest -q
	uv run --frozen ruff check src/moederncolbert_onnx tests
	bun run typecheck
	bun run build
	bun run test

browser-install:
	bun x playwright install chromium

browser-test: browser-install
	bun run demo:build
	bun run test:browser

demo:
	bun run demo

demo-node:
	bun run demo:node

benchmark:
	uv run --frozen moedern-onnx benchmark

# Run in sequence even when make was invoked with -j.
all:
	$(MAKE) setup
	$(MAKE) download
	$(MAKE) export
	$(MAKE) quantize
	$(MAKE) verify
	$(MAKE) test
	bun run demo:build
