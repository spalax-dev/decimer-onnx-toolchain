# Pipeline: runtime (M3). Exact NumPy/Pillow port of DECIMER.pre_process:
# alpha composite -> contrast LUT -> grayscale (BGR2GRAY-equivalent) -> resize ->
# border crop -> square pad -> brightness -> tf-style gaussian resize to 512x512 ->
# EfficientNet normalization. Parity with TF is the main risk; validated by tests/parity.
