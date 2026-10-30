# Agent memory

<!-- One tagged line per durable fact NOT derivable from code, git or docs:
- VERIFIED[scope] fact BC evidence
- HYPOTHESIS[scope] claim BC observation; falsifier=`cmd`
- UNKNOWN[scope] gap BC missing evidence
[scope] = the narrowest path|module|command|condition the evidence covers, not a topic: the entry decides nothing
outside it, and recurrence inside one subsystem never widens it.
Every line carries its BC and stays within 240 chars. Replace stale lines instead of appending.
Mechanizable lessons belong in tests or .vae/VERIFY.py.
Budget 4 KiB (`vae.py doctor --repo .`); entries are injected at session start. -->
- VERIFIED[ONNX export, macOS arm64 vs Linux x86-64] graph SHA-256 differs per build platform (RoPE tables differ by 1 ulp) BC node/initializer diff of the ZIP's Linux graphs vs a Mac export 2026-10-30
- VERIFIED[fp16, int8 graphs] token vectors differ between Python ORT 1.31.1, onnxruntime-node 1.21.0 and ORT Web 1.30.0 (up to 0.15) BC probe 2026-10-30; hence the "quality" contract in contracts.py
- VERIFIED[1.5m q4] mean token cosine vs FP32 0.954 to 0.961 (block 16/32) fails the 0.98 gate BC probe 2026-10-30; not shipped
