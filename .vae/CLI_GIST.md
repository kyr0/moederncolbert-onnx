# CLI gist

<!-- Commands future agents need that the Makefile verbs do not cover (setup, env, data, deploy):
- VERIFIED[purpose] `cmd`: only after an observed success
- UNKNOWN[purpose] BC gap
Budget 2 KiB (`vae.py doctor --repo .`); entries are injected at session start. -->
- VERIFIED[Claude Code sandbox] export `UV_CACHE_DIR=$TMPDIR/uv-cache UV_PYTHON_INSTALL_DIR=$TMPDIR/uv-python HF_HOME=$TMPDIR/hf-home` before uv/pytest: the sandbox denies ~/.cache writes (uv cache, transformers' trust_remote_code module cache)
- VERIFIED[Claude Code sandbox] `make e2e`/`browser-test` need local port binding and ~/Library/Caches/ms-playwright: run outside the sandbox; Python urllib downloads from HF fail there with IncompleteRead but work outside
- VERIFIED[this Mac, 2026-10-31] `bunx playwright install chromium` (build 1193) stalled while extracting; `.env` sets PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH to the installed chrome-headless-shell 1243 instead
