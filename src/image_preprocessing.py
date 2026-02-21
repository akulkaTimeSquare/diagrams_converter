"""
Image preprocessing before feeding into Qwen2.5-VL for better diagram reading.

Uses Pillow only. Automatic logic:
- Upscale images with short side < 512px to ~1024px.
- Apply light contrast enhancement when brightness stddev < 40.
"""
import os
import tempfile
from pathlib import Path
from PIL import Image, ImageEnhance, ImageStat


MIN_SHORT_SIDE = 512
TARGET_SHORT_SIDE = 1024
LOW_CONTRAST_STD = 40
CONTRAST_FACTOR = 1.2
MAX_LONG_SIDE = 1024


def preprocess_for_vlm(image_path: Path, *, enabled: bool = True) -> Path:
    """
    Apply preprocessing to the image for VLM when necessary.

    Args:
        image_path: Path to the source image.
        enabled: If False, return original path without changes.

    Returns:
        Path to the image to feed into VLM (original or temporary preprocessed file).
        When preprocessing is applied, caller is responsible for cleanup.
    """
    if not enabled:
        return image_path

    path = Path(image_path)
    if not path.exists():
        return path

    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return path

    w, h = img.size
    short_side = min(w, h)
    long_side = max(w, h)
    max_side = MAX_LONG_SIDE

    if long_side > max_side:
        scale = max_side / long_side
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        w, h = img.size
        short_side = min(w, h)
        long_side = max(w, h)

    need_upscale = short_side < MIN_SHORT_SIDE
    need_contrast = False
    if not need_upscale:
        stat = ImageStat.Stat(img.convert("L"))
        need_contrast = (stat.stddev[0] or 0) < LOW_CONTRAST_STD

    if not need_upscale and not need_contrast:
        return path

    if need_upscale:
        target_short = min(TARGET_SHORT_SIDE, max_side) if max_side else TARGET_SHORT_SIDE
        scale = target_short / short_side
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if need_contrast:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(CONTRAST_FACTOR)

    fd, out_path = tempfile.mkstemp(suffix=".png")
    try:
        os.close(fd)
        out = Path(out_path)
        img.save(out, "PNG")
        return out
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass
        return path
