# Pipeline: runtime (M3) / export (M1 freeze). Load vocab.json, map token ids <-> strings,
# apply DECIMER digit remapping (! $ ^ < > ? £ ¢ € §), and enforce start/end/pad/unk handling.
# Vocab/embedding off-by-one is audited during M1.
