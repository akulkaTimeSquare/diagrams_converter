"""
Генерация диаграммы из текста алгоритма.
Текст → общая VLM (Qwen2.5-VL) → PlantUML activity → PNG.
"""
from pathlib import Path
from typing import Literal

from src.diagram_extractor import (
    _detect_backend,
    _generate_text_with_llama_cpp_vlm,
    _generate_text_with_transformers_vlm,
    _resolve_llama_paths,
)

GENERATE_PROMPT = """По описанию алгоритма сгенерируй только код PlantUML для activity-диаграммы.
Правила:
- Выводи ТОЛЬКО код PlantUML, без markdown, без пояснений.
- Начало: @startuml, конец: @enduml.
- Тип: activity diagram (не use case, не class).
- Шаги: :Текст шага;
- Ветвления: if (условие?) then (да) else (нет) endif, или split.
- Старт: start, стоп: stop.
- Сохраняй формулировки из текста алгоритма дословно где возможно.
- Язык подписей: русский."""


def _postprocess_plantuml_response(response: str) -> str:
    """Extract and normalize PlantUML code from model output."""
    response = response.strip()
    for prefix in ("```puml", "```plantuml", "```"):
        if response.lower().startswith(prefix):
            response = response.split("\n", 1)[-1]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            break
    low = response.lower()
    if "@startuml" in low:
        start = low.index("@startuml")
        end = low.rindex("@enduml") + len("@enduml") if "@enduml" in low else len(response)
        response = response[start:end]
    if not response.strip().startswith("@"):
        response = "@startuml\n" + response + "\n@enduml"
    return response


def generate_plantuml_from_algorithm(
    algorithm_text: str,
    use_gpu: bool = False,
    max_tokens: int = 1024,
) -> str:
    """
    Сгенерировать исходный код PlantUML (activity) по тексту алгоритма.
    Использует общую загруженную VLM (Qwen2.5-VL) в режиме текст → текст.
    """
    backend = _detect_backend()
    messages = [
        {"role": "system", "content": GENERATE_PROMPT},
        {"role": "user", "content": f"Алгоритм:\n\n{algorithm_text}"},
    ]
    if backend == "transformers":
        response = _generate_text_with_transformers_vlm(messages, use_gpu, max_tokens)
    else:
        resolved_model, resolved_mmproj = _resolve_llama_paths(None, None)
        response = _generate_text_with_llama_cpp_vlm(
            messages, use_gpu, max_tokens, resolved_model, resolved_mmproj
        )
    return _postprocess_plantuml_response(response)


def generate_diagram(
    algorithm_text: str,
    output_format: Literal["png", "puml"] = "png",
    use_gpu: bool = False,
    max_tokens: int = 1024,
) -> tuple[str, Path | None]:
    """
    По тексту алгоритма сгенерировать диаграмму.

    Returns:
        (plantuml_source, png_path): исходник PlantUML и путь к PNG (если output_format=="png" и рендер удался).
    """
    from src.diagram_formats import render_plantuml_from_string

    if not (algorithm_text or algorithm_text.strip()):
        raise ValueError("Текст алгоритма не может быть пустым")

    plantuml_source = generate_plantuml_from_algorithm(
        algorithm_text.strip(),
        use_gpu=use_gpu,
        max_tokens=max_tokens,
    )

    png_path = None
    if output_format == "png":
        png_path = render_plantuml_from_string(plantuml_source)

    return plantuml_source, png_path
