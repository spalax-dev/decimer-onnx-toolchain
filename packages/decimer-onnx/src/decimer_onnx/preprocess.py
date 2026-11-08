"""Image preprocessing for DECIMER ONNX runtime (pipeline: runtime/M3, no TF/cv2).

Ports DECIMER.pre_process to NumPy/Pillow. The TF gaussian resize and the upstream
BGR2GRAY-on-RGB quirk are replicated as closely as practical; validated by SMILES parity
against the reference pipeline on fixtures.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

ArrayLike = Union[str, Path, np.ndarray, Image.Image]

# ImageNet normalization (matches efficientnet.tfkeras.preprocess_input behavior observed).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _open_rgba(image: ArrayLike) -> Image.Image:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGBA")
        if image.shape[2] == 4:
            return Image.fromarray(image.astype(np.uint8), mode="RGBA")
        return Image.fromarray(image.astype(np.uint8)).convert("RGBA")
    if isinstance(image, Image.Image):
        return image.convert("RGBA")
    return Image.open(image).convert("RGBA")


def remove_transparent(image: ArrayLike) -> Image.Image:
    png = _open_rgba(image)
    background = Image.new("RGBA", png.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, png)


def increase_contrast(image: Image.Image) -> Image.Image:
    arr = np.asarray(image)
    min_val = int(np.min(arr))
    max_val = int(np.max(arr))
    lut = np.zeros(256, dtype=np.uint8)
    if max_val >= min_val:
        lut[min_val : max_val + 1] = np.linspace(0, 255, max_val - min_val + 1, dtype=np.uint8)
    return Image.fromarray(lut[arr])


def get_bnw_image(image: Image.Image) -> Image.Image:
    # Upstream uses cv2.cvtColor(im_np, cv2.COLOR_BGR2GRAY) on an RGB PIL array,
    # effectively swapping R/B weights (0.114*R + 0.587*G + 0.299*B). Replicate quirk.
    im = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = 0.114 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.299 * im[:, :, 2]
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    im_pil = Image.fromarray(gray, mode="L")
    return ImageEnhance.Contrast(im_pil).enhance(1.8)


def resize_byratio(image: Image.Image) -> Image.Image:
    maxwidth = 512
    ratio = maxwidth / max(image.width, image.height)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    return image.resize(new_size, resample=Image.Resampling.LANCZOS)


def get_resize(image: Image.Image) -> Image.Image:
    w, h = image.size
    if (w == h) and (w < 512):
        return image.resize((512, 512), resample=Image.Resampling.LANCZOS)
    if w >= 512 or h >= 512:
        return image
    return resize_byratio(image)


def delete_empty_borders(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("L"))
    mask = arr > 200
    rows = np.flatnonzero((~mask).sum(axis=1))
    cols = np.flatnonzero((~mask).sum(axis=0))
    if rows.size == 0 or cols.size == 0:
        return image
    crop = arr[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    return Image.fromarray(crop)


def central_square_image(image: Image.Image) -> Image.Image:
    max_wh = int(1.2 * max(image.size))
    if max_wh < 512:
        max_wh = 512
    new_im = Image.new("L", (max_wh, max_wh), 255)
    paste_pos = ((new_im.size[0] - image.size[0]) // 2, (new_im.size[1] - image.size[1]) // 2)
    new_im.paste(image, paste_pos)
    return new_im


def increase_brightness(image: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(1.6)


def _resize_to_512(img: Image.Image) -> np.ndarray:
    # Approximate TF gaussian+antialias resize. PIL LANCZOS is a close antialiased resampler;
    # SMILES-level parity is the acceptance criterion.
    rgb = img.convert("RGB").resize((512, 512), resample=Image.Resampling.LANCZOS)
    return np.asarray(rgb, dtype=np.float32)


def preprocess_image(image: ArrayLike) -> np.ndarray:
    img = remove_transparent(image)
    img = increase_contrast(img)
    img = get_bnw_image(img)
    img = get_resize(img)
    img = delete_empty_borders(img)
    img = central_square_image(img)
    img = increase_brightness(img)
    arr = _resize_to_512(img)
    arr = (arr / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr.astype(np.float32)
