# Pipeline: stage 3 — FP32 export (M2). Build clean tf.function wrappers:
# encoder(image[1,512,512,3]) -> enc[1,256,512]; decoder(tokens[1,T], enc, mask) -> probs.
# Keeps batch fixed, exposes dynamic sequence length, and avoids the training/loop graph.
