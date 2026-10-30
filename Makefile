# Project interface (defuss-vae layout): the same verbs for humans, agents and CI.
# Service: the built demo (vite preview); logs → var/log/$(NAME).*, pid → tmp/$(NAME).pid (both gitignored).
# Models: MODELS selects checkpoints; each builds into output/hf/<repo name> (the Hugging Face upload folder).
# Optional overrides from .env (see .env.example), exported to every recipe.
-include .env
NAME   ?= demo
# Not Vite's default 4173, which other local previews commonly hold.
PORT   ?= 4791
DEMO_URL ?= http://127.0.0.1:$(PORT)
export DEMO_URL PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
MODELS ?= 1.5m 27m
# CI passes --with-deps for a minimal Linux host.
PLAYWRIGHT_FLAGS ?=
CLI     = uv run --frozen moedern-onnx
# CPU binaries ship inside onnxruntime-node; skip its optional CUDA download.
export ONNXRUNTIME_NODE_INSTALL = skip
export PATH := $(PATH):$(HOME)/.local/bin:$(HOME)/.bun/bin
LOG   = var/log/$(NAME)
PID   = tmp/$(NAME).pid
ALIVE = [ -f $(PID) ] && kill -0 "$$(cat $(PID))" 2>/dev/null

.PHONY: setup models download export quantize parity reference demo-build browser-test demo demo-node \
	start stop restart status log metrics bench test coverage lint e2e verify

# Fresh clone: locked deps, Chromium, the JS library and both models' upload folders.
setup:
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	@command -v bun >/dev/null 2>&1 || curl -fsSL https://bun.sh/install | bash
	uv sync --frozen --python 3.11
	bun install --frozen-lockfile
	bunx playwright install $(PLAYWRIGHT_FLAGS) chromium
	bun run build
	$(MAKE) models

# Per model: pinned download (hash-checked), FP32 export, variants, parity gates, model card.
models:
	@for m in $(MODELS); do $(CLI) build --model $$m || exit 1; done
download export quantize parity reference:
	@for m in $(MODELS); do $(CLI) $@ --model $$m || exit 1; done

demo-build: ; bun run build && bun run demo:build
# Interactive demo with hot reload; `make start` serves the production build instead.
demo: ; bun run demo
demo-node: ; node examples/node.mjs output/hf/SauerkrautLM-MoEdernColBERT-1.5M-ONNX q8
browser-test: demo-build
	$(MAKE) start
	node scripts/verify-browser.mjs; s=$$?; $(MAKE) stop; exit $$s

start:
	@mkdir -p var/log tmp
	@if $(ALIVE); then echo "$(NAME) running pid=$$(cat $(PID))"; exit 0; fi; \
	[ -d web-dist ] || { echo "web-dist/ missing: run make demo-build"; exit 1; }; \
	nohup node node_modules/vite/bin/vite.js preview --host 127.0.0.1 --port $(PORT) --strictPort \
		>>$(LOG).stdout 2>>$(LOG).stderr & echo $$! >$(PID); \
	for i in $$(seq 100); do $(ALIVE) || break; \
		curl -fs -o /dev/null http://127.0.0.1:$(PORT)/models/index.json && { echo "$(NAME) http://127.0.0.1:$(PORT)/ pid=$$(cat $(PID))"; exit 0; }; \
		sleep 0.1; done; rm -f $(PID); echo "$(NAME) failed to start; see $(LOG).stderr"; exit 1
stop:
	@if $(ALIVE); then kill -TERM "$$(cat $(PID))"; echo "$(NAME) stopped"; else echo "$(NAME) not running"; fi; rm -f $(PID)
restart: ; $(MAKE) stop && $(MAKE) start
status: ; @if $(ALIVE); then echo "$(NAME) running pid=$$(cat $(PID)) http://127.0.0.1:$(PORT)/"; else echo "$(NAME) stopped"; exit 3; fi
log: ; @tail -n 50 $(LOG).stdout $(LOG).stderr 2>/dev/null || echo "no logs in var/log"
# Download size per graph of every built upload folder.
metrics: ; @for f in output/hf/*/model_*.onnx; do printf '%12s %s\n' "$$(wc -c <$$f | tr -d ' ')" "$$f"; done
bench: ; @for m in $(MODELS); do $(CLI) bench --model $$m || exit 1; done

# Python suite builds both models into a temp dir; the JS suite reads output/hf (make models).
test:
	uv run --frozen pytest -q
	bun run build
	node --test tests-js/*.test.mjs
	node scripts/verify-node.mjs
coverage: ; @mkdir -p tmp; uv run --frozen pytest -q --cov=moederncolbert_onnx --cov-report=term >tmp/coverage.txt 2>&1; s=$$?; \
	grep -E '^TOTAL' tmp/coverage.txt; exit $$s
lint:
	uv run --frozen ruff check .
	bunx oxlint --deny-warnings
	bunx tsc --noEmit

# Consumes the publishable artifacts like a user: the wheel's CLI rebuilds both upload folders and runs
# every command; a clean project installs the packed npm tarball and reranks input/ with every folder and
# precision; Chromium drives the built demo served by `make start`.
WHEEL_CLI = uv run --isolated --no-project --with "$$(ls output/wheel/*.whl)" moedern-onnx
e2e:
	rm -rf output/wheel output/e2e tmp/e2e-consumer
	mkdir -p output/e2e tmp/e2e-consumer
	uv build --wheel --out-dir output/wheel
	$(WHEEL_CLI) download --model 1.5m
	$(WHEEL_CLI) export --model 1.5m
	$(WHEEL_CLI) quantize --model 1.5m
	$(WHEEL_CLI) parity --model 1.5m
	$(WHEEL_CLI) build --model 27m
	$(WHEEL_CLI) reference --model 1.5m --out output/e2e/reference-1.5m.json
	$(WHEEL_CLI) demo --model 27m --precision q4 > output/e2e/demo-27m-q4.json
	$(WHEEL_CLI) bench --model 1.5m --repeats 2 --reports output/e2e
	bun run build
	bun pm pack --destination output/e2e
	cp scripts/e2e-consumer.mjs tmp/e2e-consumer/consumer.mjs
	cd tmp/e2e-consumer && echo '{"name":"e2e-consumer","private":true,"type":"module"}' >package.json \
		&& bun add "$$(ls $(CURDIR)/output/e2e/*.tgz)" && node consumer.mjs $(CURDIR)
	$(MAKE) browser-test

# What CI runs; the gate runs the same verbs one by one and fails while verify skips any.
verify: lint test coverage e2e
