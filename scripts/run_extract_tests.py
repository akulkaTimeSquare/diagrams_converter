#!/usr/bin/env python3
"""
Run extract API tests against test images and compare with expected outputs from test.txt.
Usage: python scripts/run_extract_tests.py [--api-url URL] [--max-tokens N]
Requires: requests (pip install requests)
"""
import argparse
import io
import sys

# Fix Windows console encoding for Cyrillic output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import re
import sys
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import requests
except ImportError:
    print("Error: requests not installed. Run: pip install requests")
    sys.exit(1)


def parse_test_txt(test_txt_path: Path) -> dict[str, str]:
    """Parse test.txt into {filename: expected_output}."""
    content = test_txt_path.read_text(encoding="utf-8")
    blocks = {}
    current_file = None
    lines: list[str] = []

    for line in content.splitlines():
        stripped = line.strip()
        # New block: line is exactly a .png filename
        if re.match(r"^\d+\.png$", stripped):
            if current_file is not None:
                blocks[current_file] = "\n".join(lines).strip()
            current_file = stripped
            lines = []
        elif current_file is not None:
            lines.append(line)

    if current_file is not None:
        blocks[current_file] = "\n".join(lines).strip()

    return blocks


def normalize_text(text: str) -> str:
    """Normalize text for comparison: collapse whitespace, trim."""
    if not text:
        return ""
    # Normalize multiple spaces/tabs to single space
    normalized = re.sub(r"[ \t]+", " ", text)
    # Normalize line endings
    normalized = re.sub(r"\r\n", "\n", normalized)
    normalized = re.sub(r"\r", "\n", normalized)
    # Trim each line and remove empty lines at start/end
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    return "\n".join(lines)


def extract_steps(text: str) -> list[str]:
    """Extract numbered steps (N. ...) from output for partial comparison."""
    steps = []
    for line in text.splitlines():
        m = re.match(r"^(\d+)\.\s*(.+)$", line.strip())
        if m:
            steps.append(m.group(2).strip())
    return steps


def step_similarity(expected_steps: list[str], actual_steps: list[str]) -> float:
    """Compute similarity: fraction of expected steps that have a matching actual step."""
    if not expected_steps:
        return 1.0 if not actual_steps else 0.0
    matches = 0
    for exp in expected_steps:
        for act in actual_steps:
            # Normalize for comparison: lower, collapse spaces
            en = re.sub(r"\s+", " ", exp.lower().strip())
            an = re.sub(r"\s+", " ", act.lower().strip())
            if en in an or an in en or en == an:
                matches += 1
                break
    return matches / len(expected_steps)


def call_extract_api(image_path: Path, api_url: str, max_tokens: int, use_gpu: bool, preprocess: bool) -> str:
    """POST image to /extract and return algorithm text."""
    url = f"{api_url.rstrip('/')}/extract"
    with open(image_path, "rb") as f:
        resp = requests.post(
            url,
            files={"file": (image_path.name, f, "image/png")},
            data={
                "use_gpu": str(use_gpu).lower(),
                "max_tokens": max_tokens,
                "preprocess": str(preprocess).lower(),
            },
            timeout=120,
        )
    resp.raise_for_status()
    data = resp.json()
    return (data.get("algorithm") or "").strip()


def call_extract_direct(image_path: Path, max_tokens: int, use_gpu: bool, preprocess: bool) -> str:
    """Call extract_algorithm directly (bypasses API, uses current prompt from diagram_extractor)."""
    from src.diagram_extractor import extract_algorithm

    return extract_algorithm(
        image_path,
        use_gpu=use_gpu,
        max_tokens=max_tokens,
        use_preprocessing=preprocess,
        log_timings=False,
    ).strip()


def main():
    parser = argparse.ArgumentParser(description="Run extract API tests against test images")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--max-tokens", type=int, default=1024, help="max_tokens for extract")
    parser.add_argument("--use-gpu", action="store_true", help="use GPU for VLM")
    parser.add_argument("--no-preprocess", action="store_true", help="disable image preprocessing")
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "test", help="test directory")
    parser.add_argument("--direct", action="store_true", help="call extract_algorithm directly (no API)")
    args = parser.parse_args()

    test_txt = args.test_dir / "test.txt"
    if not test_txt.exists():
        print(f"Error: {test_txt} not found")
        sys.exit(1)

    expected = parse_test_txt(test_txt)
    if not expected:
        print("Error: no test cases parsed from test.txt")
        sys.exit(1)

    print(f"Parsed {len(expected)} test cases from test.txt")
    print(f"API: {args.api_url} | max_tokens={args.max_tokens} | preprocess={not args.no_preprocess}\n")

    exact_pass = 0
    partial_pass = 0
    failures: list[tuple[str, str, str]] = []

    for filename in sorted(expected.keys()):
        image_path = args.test_dir / filename
        if not image_path.exists():
            print(f"SKIP {filename}: image not found")
            continue

        exp_text = expected[filename]
        exp_norm = normalize_text(exp_text)
        exp_steps = extract_steps(exp_text)

        try:
            if args.direct:
                actual = call_extract_direct(
                    image_path, args.max_tokens, args.use_gpu, not args.no_preprocess
                )
            else:
                actual = call_extract_api(
                    image_path, args.api_url, args.max_tokens, args.use_gpu, not args.no_preprocess
                )
        except (requests.RequestException, Exception) as e:
            print(f"FAIL {filename}: error - {e}")
            failures.append((filename, exp_text, f"API error: {e}"))
            continue

        act_norm = normalize_text(actual)
        act_steps = extract_steps(actual)

        exact = exp_norm == act_norm
        similarity = step_similarity(exp_steps, act_steps)
        partial = similarity >= 0.8

        if exact:
            exact_pass += 1
            print(f"PASS (exact)  {filename}")
        elif partial:
            partial_pass += 1
            print(f"PASS (partial, {similarity:.0%}) {filename}")
        else:
            print(f"FAIL {filename} (similarity {similarity:.0%})")
            failures.append((filename, exp_text, actual))

    total = len(expected)
    print("\n" + "=" * 60)
    print(f"Results: {exact_pass} exact, {partial_pass} partial, {total - exact_pass - partial_pass} fail (of {total})")

    if failures:
        print("\n--- Failures ---")
        for fname, exp, act in failures:
            print(f"\n### {fname} ###")
            print("Expected:")
            print(exp[:500] + ("..." if len(exp) > 500 else ""))
            print("Actual:")
            print(act[:500] + ("..." if len(act) > 500 else ""))
            print("-" * 40)

    # Exit 0 if 8+ pass (exact or partial)
    passed = exact_pass + partial_pass
    sys.exit(0 if passed >= 8 else 1)


if __name__ == "__main__":
    main()
