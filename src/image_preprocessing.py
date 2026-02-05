"""
Препроцессинг изображений перед подачей в Qwen2.5-VL для лучшего чтения диаграмм.
Используется только Pillow. Авто-логика: апскейл мелких, лёгкое улучшение контраста при необходимости.
"""
import tempfile
from pathlib import Path

from PIL import Image, ImageEnhance, ImageStat


MIN_SHORT_SIDE = 512
TARGET_SHORT_SIDE = 1024
LOW_CONTRAST_STD = 40
CONTRAST_FACTOR = 1.2


def preprocess_for_vlm(image_path: Path, *, enabled: bool = True) -> Path:
    """
    При необходимости применить препроцессинг к изображению для VLM.

    - enabled=False: вернуть исходный путь без изменений.
    - Короткая сторона < 512px: апскейл до ~1024px.
    - Низкий контраст (std яркости < 40): лёгкое улучшение контраста.
    - Иначе: вернуть исходный путь.

    Возвращает путь к изображению для подачи в VLM. Если создан временный файл,
    вызывающий код должен удалить его после использования.
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
    need_upscale = short_side < MIN_SHORT_SIDE
    need_contrast = False
    if not need_upscale:
        stat = ImageStat.Stat(img.convert("L"))
        need_contrast = (stat.stddev[0] or 0) < LOW_CONTRAST_STD

    if not need_upscale and not need_contrast:
        return path

    if need_upscale:
        scale = TARGET_SHORT_SIDE / short_side
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if need_contrast:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(CONTRAST_FACTOR)

    fd, out_path = tempfile.mkstemp(suffix=".png")
    try:
        import os
        os.close(fd)
        out = Path(out_path)
        img.save(out, "PNG")
        return out
    except Exception:
        import os
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass
        return path
