"""
Generate diagrams from algorithm text.

Pipeline: text -> VLM (Qwen2.5-VL) -> PlantUML activity diagram -> PNG.
Reuses the same VLM instance as diagram extraction.
"""
import logging
import os
import re
import time
from pathlib import Path
from typing import Literal

from src.diagram_extractor import (
    _detect_backend,
    _generate_text_with_llama_cpp_vlm,
    _generate_text_with_transformers_vlm,
    _resolve_llama_paths,
)
from src.diagram_formats import render_plantuml_from_string
from src.diagram_prompts import GENERATE_PROMPT

logger = logging.getLogger(__name__)


# PlantUML post-processing

def _strip_markdown_code_blocks(text: str) -> str:
    """Remove markdown code block wrappers (```puml, ```plantuml, ```)."""
    text = text.strip()
    for prefix in ("```puml", "```plantuml", "```"):
        if text.lower().startswith(prefix):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            break
    return text


def _normalize_plantuml_bounds(text: str) -> str:
    """Ensure @startuml/@enduml wrapper; extract if embedded in larger output."""
    # Fix common typo: model sometimes writes :enduml instead of @enduml
    text = text.replace(":enduml", "@enduml")
    low = text.lower()
    if "@startuml" in low:
        start = low.index("@startuml")
        end = low.rindex("@enduml") + len("@enduml") if "@enduml" in low else len(text)
        text = text[start:end]
    if not text.strip().startswith("@"):
        text = "@startuml\n" + text + "\n@enduml"
    # Add stop before @enduml if missing (PlantUML requires it)
    if "@enduml" in text.lower() and not re.search(r"(?:^|\n)\s*stop\s*(?:\n|$)", text, re.IGNORECASE):
        text = text.replace("@enduml", "stop\n@enduml")
    return text


def _postprocess_plantuml_response(response: str) -> str:
    """Extract and normalize PlantUML code from model output."""
    text = _strip_markdown_code_blocks(response)
    text = _normalize_plantuml_bounds(text)
    return text


# VLM backends

def _generate_text_with_llamacpp_server(messages: list[dict], max_tokens: int) -> str:
    """Generate text using an external llama.cpp server (OpenAI-compatible API)."""
    import requests

    url = os.environ.get("LLAMACPP_URL", "http://localhost:8080/v1/chat/completions")
    model_name = os.environ.get("LLAMACPP_MODEL", "llama.cpp")
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("llama.cpp server returned empty choices")
    return choices[0]["message"]["content"].strip()


def _call_vlm_for_generation(
    messages: list[dict],
    use_gpu: bool,
    max_tokens: int,
) -> tuple[str, str]:
    """
    Call the appropriate VLM backend for text generation.

    Returns: (response_text, backend_label).
    """
    backend = _detect_backend()
    backend_env = os.environ.get("LLM_BACKEND", "").strip().lower()

    if backend_env == "llamacpp":
        response = _generate_text_with_llamacpp_server(messages, max_tokens)
        return response, "llamacpp_server"

    if backend == "transformers":
        response = _generate_text_with_transformers_vlm(messages, use_gpu, max_tokens)
        return response, "transformers"

    resolved_model, resolved_mmproj = _resolve_llama_paths(None, None)
    response = _generate_text_with_llama_cpp_vlm(
        messages, use_gpu, max_tokens, resolved_model, resolved_mmproj
    )
    return response, "llama_cpp"


# Main entry points

def _build_generation_messages(algorithm_text: str) -> list[dict]:
    """Build chat messages for PlantUML generation."""
    return [
        {"role": "system", "content": GENERATE_PROMPT},
        {"role": "user", "content": f"Алгоритм:\n\n{algorithm_text}"},
    ]


def generate_plantuml_from_algorithm(
    algorithm_text: str,
    use_gpu: bool = False,
    max_tokens: int = 1024,
) -> str:
    """
    Generate PlantUML (activity) source code from algorithm text.

    Uses the shared VLM (Qwen2.5-VL) in text-to-text mode.
    Backend is selected automatically: transformers, llama_cpp, or llamacpp server.
    """
    messages = _build_generation_messages(algorithm_text)
    start = time.perf_counter()

    response, backend_label = _call_vlm_for_generation(messages, use_gpu, max_tokens)

    elapsed = time.perf_counter() - start
    logger.info("timings generate: backend=%s total=%.4fs", backend_label, elapsed)

    return _postprocess_plantuml_response(response)


def generate_diagram(
    algorithm_text: str,
    output_format: Literal["png", "puml"] = "png",
    use_gpu: bool = False,
    max_tokens: int = 1024,
) -> tuple[str, Path | None]:
    """
    Generate a diagram from algorithm text.

    Args:
        algorithm_text: Process description or list of steps.
        output_format: "puml" for PlantUML code only; "png" to also render to PNG.
        use_gpu: Use GPU for VLM inference.
        max_tokens: Maximum tokens for PlantUML generation.

    Returns:
        Tuple of (plantuml_source, png_path). png_path is None when output_format=="puml"
        or when PlantUML render fails.
    """
    if not (algorithm_text or algorithm_text.strip()):
        raise ValueError("Algorithm text cannot be empty")

    plantuml_source = generate_plantuml_from_algorithm(
        algorithm_text.strip(),
        use_gpu=use_gpu,
        max_tokens=max_tokens,
    )

    png_path = None
    if output_format == "png":
        png_path = render_plantuml_from_string(plantuml_source)

    return plantuml_source, png_path
