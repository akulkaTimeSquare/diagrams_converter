"""Test VLM-based algorithm extraction from diagram images."""
import io
import sys
import time
from pathlib import Path

# Fix Windows console encoding for Cyrillic output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.diagram_extractor import extract_algorithm


def main() -> None:
    """Run extract_algorithm on a test image (PNG or SVG)."""
    # Prefer PNG; fall back to SVG (extract_algorithm converts it)
    image_path = ROOT / "data" / "test" / "1.png"
    if not image_path.exists():
        image_path = ROOT / "data" / "test" / "diagram.svg"
    if not image_path.exists():
        print("No test image found in data/test/ (1.png or diagram.svg)")
        return

    start = time.perf_counter()
    result = extract_algorithm(image_path, use_gpu=False, max_tokens=512)
    elapsed = time.perf_counter() - start

    print(f"Result ({elapsed:.1f}s):")
    print(result)


if __name__ == "__main__":
    main()
