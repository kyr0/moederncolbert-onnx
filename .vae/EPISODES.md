# Episodes

<!-- Newest last. The gate appends FAIL|DONE|FINDING; past 100 entries it trims only its own oldest DONE|FAIL|learned
FINDING (git keeps them). Leads (LESSON, FINDING learn=none) stay until wrap settles them; doctor fails above 30.
Agents append `<UTC ISO> s=<session> LESSON <VAE-DIALECT>` only for a falsified HYPOTHESIS, a dead end, or a root cause.
Session start injects the 3 newest open entries: LESSON, FINDING learn=none, FAIL without a later DONE.
Wrap: EVERY lead becomes a test | .vae/VERIFY.py rule | one MEMORY line with BC, then delete; OR delete BC evidence. -->

2026-10-30T23:30:52Z s=8e62adc6 FAIL prose,tests.unit
2026-10-30T23:45:52Z s=8e62adc6 DONE fp=b835072ec5bf cov=90.0% paths=.github/workflows/verify.yml,AGENTS.md,ARCH.md,Makefile(+44)
2026-10-30T23:45:52Z s=8e62adc6 FINDING src/moederncolbert_onnx/quantization.py:build fp16 learn=test: graph_info runs onnx.checker on every built graph; tests/conftest.py builds fp16 for both models
2026-10-30T23:45:52Z s=8e62adc6 FINDING Makefile:start learn=memory: probe now requires our pid alive and our /models/index.json; default port 4791; CLI_GIST records the collision
2026-10-30T23:45:52Z s=8e62adc6 FINDING tests-js/contract.test.mjs:folders learn=none: fixed at the one unfiltered caller; reproduced by touching output/hf/.DS_Store, suite passes 14/14
2026-10-30T23:45:52Z s=8e62adc6 FINDING scripts/verify-browser.mjs:reports learn=none: timings stay in ignored output/browser/report.json; python and node reports were checked byte-stable across a rebuild
2026-10-30T23:45:52Z s=8e62adc6 FINDING reports/*.json (top level) learn=none: deleted
2026-10-30T23:45:52Z s=8e62adc6 FINDING ARCH.md:q8 bullet B01 learn=none: replaced by the measured cross-runtime q8 bound
2026-10-30T23:45:52Z s=8e62adc6 FINDING AGENTS.md:5-14 T06 learn=verifier: CONFIG['prose']['allow'] lists the arrow for AGENTS.md only
2026-10-30T23:48:35Z s=8e62adc6 DONE fp=c33a20048ee2 cov=90.0% paths=.github/workflows/verify.yml,AGENTS.md,ARCH.md,Makefile(+38)
