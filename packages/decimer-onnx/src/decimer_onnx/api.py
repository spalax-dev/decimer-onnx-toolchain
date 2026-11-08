# Pipeline: runtime (M3). Public API: DecimerOCSR.from_path(...).predict(image).
# Loads encoder.onnx + decoder.onnx + vocab.json, runs preprocessing and greedy decode,
# returns SMILES + per-token confidence + mean + incomplete. No TensorFlow dependency.
