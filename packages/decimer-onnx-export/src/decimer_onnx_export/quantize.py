# Pipeline: stage 4 — Quantize (M4). Build candidates E1 (dynamic INT8), E2 (static INT8 QDQ),
# E3 (static encoder + dynamic decoder), E4 (FP16), and rescue E5 (INT16 QDQ) only if needed.
# Uses ORT quantization APIs; records per-variant config and artifact hashes.
