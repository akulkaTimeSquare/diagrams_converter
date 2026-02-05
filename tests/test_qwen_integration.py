"""
Test Qwen2.5-VL integration: extract algorithm from diagram image.
Run from project root: python tests/test_qwen_integration.py
"""
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.diagram_extractor import extract_algorithm


def main() -> None:
    # Use a sample image from Picture/
    image_path = ROOT / "Picture" / "1.png"
    if not image_path.exists():
        image_path = ROOT / "test" / "3.png"
    if not image_path.exists():
        print("No test image found in Picture/ or test/")
        return

    print(f"Testing with image: {image_path}")
    print("Loading model and running inference...")

    start = time.perf_counter()
    result = extract_algorithm(image_path, use_gpu=False, max_tokens=1024)
    elapsed = time.perf_counter() - start

    print(f"\n--- Result ({elapsed:.1f}s) ---")
    print(result)
    print("---")

    if elapsed > 20:
        print(f"Warning: inference took {elapsed:.1f}s (target < 20s)")


if __name__ == "__main__":
    main()
