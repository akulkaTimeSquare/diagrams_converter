"""
Extract algorithm/process description from diagram images using Qwen2.5-VL-3B.
Supports: Transformers (primary), llama-cpp-python (optional, when available).
Single VLM instance is cached and reused for both extract (image→text) and generate (text→PlantUML).
"""
import os
import threading
from pathlib import Path
from typing import Any, Optional, Union

# Serialize access to the single VLM instance (no concurrent inference).
_VLM_LOCK = threading.Lock()

# Пути по умолчанию для llama.cpp (каталог проекта/models/)
_DEFAULT_LLAMA_MODEL = "Qwen2.5-VL-3B-Instruct-q4_k_m.gguf"
_DEFAULT_LLAMA_MMPROJ = "Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf"

# Промпт для извлечения алгоритма в формате, совпадающем с test.txt
DIAGRAM_PROMPT = """Извлеки алгоритм или бизнес-процесс с этой диаграммы. Ответь только на русском языке.

Роли — только если есть дорожки на диаграмме:
- Дорожки (swimlanes/lanes) = отдельные горизонтальные или вертикальные полосы, у каждой есть подпись с названием роли (например "Инициатор", "Координатор") прямо на диаграмме.
- Если таких полос с подписями НЕТ — это диаграмма без ролей. Запрещено: придумывать роли, писать "| Роль", "Инициатор", "Координатор", "Руководитель" и т.п. Вывод только: первая строка "Шаг", далее "1. Текст", "2. Текст", ... — без символа "|" и без любых названий ролей.
- Если дорожки ЕСТЬ — используй формат с колонкой "Роль" (см. ниже). Роль берётся только из подписи дорожки, в которой стоит фигура.

Формат при наличии дорожек (BPMN с ролями):
- Первая строка: "Шаг" + табы/пробелы + "| Роль". Каждая строка: "N. Текст" + выравнивание + "| Роль". Роль = полное название дорожки с диаграммы. Конечные события не выводи как шаг.

Формат без дорожек (нет ролей):
- Первая строка: "Шаг". Далее: "1. Текст", "2. Текст", ... — без "|" и без ролей.

Без дублирования:
- Каждая фигура (прямоугольник задачи, ромб решения, событие) входит в список ровно один раз. Не повторяй один и тот же шаг с разными номерами. Количество пунктов = количеству узлов на диаграмме (минус конечные кружки при необходимости).

Ветвления (строго):
- Запрещено объединять ветки в одну строку. Нельзя писать "Да -> Договориться о графике", "Нет -> Фокус на учебе" и т.п. Каждый узел — отдельная строка: "N. Текст из фигуры" (только текст, который написан внутри фигуры). Подписи на стрелках (да, нет, Да, Нет) в текст шага не включай и не используй как префикс.
- Ромб решения — один пункт списка с текстом из ромба (например "4. Найдена?").
- Порядок после ромба: сначала пройди одну ветку до конца (все узлы по стрелкам до конечного состояния или слияния), затем вторую ветку целиком, затем общее продолжение если есть. Пример: после "Найдена?" идут "5. Договориться о графике", "6. Совмещение работы и учебы", затем "7. Фокус на учебе", "8. Только учеба" — каждая фигура отдельным пунктом, без "Да"/"Нет" в тексте шага.
- Кружки конечных состояний (например "Совмещение работы и учебы", "Только учеба") выводи как отдельные шаги, если на них есть подпись.

Текст шага — дословно с диаграммы (без замены слов). Включай только текст из фигур (прямоугольник, ромб, кружок)."""

_BACKEND: Optional[str] = None

# Cached VLM instances (singleton per backend)
_transformers_model: Any = None
_transformers_processor: Any = None
_transformers_device: Optional[str] = None
_transformers_use_gpu: Optional[bool] = None
_llama_llm: Any = None
_llama_chat_handler: Any = None
_llama_use_gpu: Optional[bool] = None
_llama_model_path: Optional[Path] = None
_llama_mmproj_path: Optional[Path] = None


def _get_transformers_vlm(use_gpu: bool) -> tuple[Any, Any, str]:
    """Load or return cached Transformers VLM (model, processor, device)."""
    global _transformers_model, _transformers_processor, _transformers_device, _transformers_use_gpu
    with _VLM_LOCK:
        if _transformers_model is not None and _transformers_use_gpu == use_gpu:
            return _transformers_model, _transformers_processor, _transformers_device
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        device = "cuda" if use_gpu else "cpu"
        _transformers_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto" if use_gpu else None,
        )
        if not use_gpu:
            _transformers_model = _transformers_model.to(device)
        _transformers_processor = AutoProcessor.from_pretrained(model_id)
        _transformers_device = device
        _transformers_use_gpu = use_gpu
        return _transformers_model, _transformers_processor, _transformers_device


def _get_llama_cpp_vlm(
    use_gpu: bool,
    model_path: Optional[Path],
    mmproj_path: Optional[Path],
    n_ctx: int = 2048,
) -> Any:
    """Load or return cached llama-cpp-python VLM (Llama instance)."""
    global _llama_llm, _llama_chat_handler, _llama_use_gpu, _llama_model_path, _llama_mmproj_path
    with _VLM_LOCK:
        if _llama_llm is not None and _llama_use_gpu == use_gpu:
            if (model_path is None and mmproj_path is None) or (
                _llama_model_path == model_path and _llama_mmproj_path == mmproj_path
            ):
                return _llama_llm
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler

        if model_path is None and mmproj_path is None:
            _llama_chat_handler = Qwen25VLChatHandler.from_pretrained(
                repo_id="Mungert/Qwen2.5-VL-3B-Instruct-GGUF",
                filename="*mmproj*",
            )
            _llama_llm = Llama.from_pretrained(
                repo_id="Mungert/Qwen2.5-VL-3B-Instruct-GGUF",
                filename="*q4_k_m*",
                chat_handler=_llama_chat_handler,
                n_ctx=n_ctx,
                n_gpu_layers=-1 if use_gpu else 0,
            )
        else:
            if model_path is None or mmproj_path is None:
                raise ValueError("Both model_path and mmproj_path must be provided for local models")
            _llama_chat_handler = Qwen25VLChatHandler(clip_model_path=str(mmproj_path))
            _llama_llm = Llama(
                model_path=str(model_path),
                chat_handler=_llama_chat_handler,
                n_ctx=n_ctx,
                n_gpu_layers=-1 if use_gpu else 0,
            )
        _llama_use_gpu = use_gpu
        _llama_model_path = model_path
        _llama_mmproj_path = mmproj_path
        return _llama_llm


def _make_placeholder_image_path() -> Path:
    """Create a minimal 1x1 PNG for text-only VLM calls (processor expects at least one image)."""
    import tempfile
    from PIL import Image
    img = Image.new("RGB", (1, 1), color=(255, 255, 255))
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.save(path)
    return Path(path)


def _generate_text_with_transformers_vlm(
    messages: list[dict],
    use_gpu: bool,
    max_tokens: int,
) -> str:
    """Generate text using cached Transformers VLM. Uses a 1x1 placeholder image so processor gets valid input."""
    from qwen_vl_utils import process_vision_info

    model, processor, device = _get_transformers_vlm(use_gpu)
    # Qwen2.5-VL processor can yield empty batch with no image; add placeholder image for text-only.
    placeholder_path = None
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, list):
            has_image = any(isinstance(c, dict) and c.get("type") == "image" for c in content)
        else:
            has_image = False
        if not has_image:
            placeholder_path = _make_placeholder_image_path()
            if isinstance(content, list):
                new_content = [{"type": "image", "image": str(placeholder_path)}] + list(content)
            else:
                new_content = [{"type": "image", "image": str(placeholder_path)}, {"type": "text", "text": content}]
            messages = [{**msg, "content": new_content} if msg.get("role") == "user" else msg for msg in messages]
        break
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        # Не передавать videos при пустом списке — иначе video_processor падает на videos[0].
        processor_kwargs = {
            "text": [text],
            "images": image_inputs if image_inputs else [],
            "padding": True,
            "return_tensors": "pt",
        }
        if video_inputs:
            processor_kwargs["videos"] = video_inputs
        inputs = processor(**processor_kwargs)
        inputs = inputs.to(device)
        with _VLM_LOCK:
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
        if generated_ids.shape[0] == 0:
            raise ValueError("VLM returned empty generation (batch size 0)")
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not output_text:
            raise ValueError("VLM returned empty decoded text")
        return output_text[0].strip()
    finally:
        if placeholder_path is not None and placeholder_path.exists():
            placeholder_path.unlink(missing_ok=True)


def _generate_text_with_llama_cpp_vlm(
    messages: list[dict],
    use_gpu: bool,
    max_tokens: int,
    model_path: Optional[Path],
    mmproj_path: Optional[Path],
    n_ctx: int = 2048,
) -> str:
    """Generate text using cached llama-cpp VLM (text-only, no image)."""
    llm = _get_llama_cpp_vlm(use_gpu, model_path, mmproj_path, n_ctx)
    with _VLM_LOCK:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
        )
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("VLM returned empty choices (no text generated)")
    return choices[0]["message"]["content"].strip()


def ensure_vlm_loaded(use_gpu: bool = False) -> None:
    """Preload the VLM so the first request does not wait. Call at API startup."""
    backend = _detect_backend()
    if backend == "transformers":
        _get_transformers_vlm(use_gpu)
    else:
        resolved_model, resolved_mmproj = _resolve_llama_paths(None, None)
        _get_llama_cpp_vlm(use_gpu, resolved_model, resolved_mmproj)


def _resolve_llama_paths(
    model_path: Optional[Path],
    mmproj_path: Optional[Path],
) -> tuple[Optional[Path], Optional[Path]]:
    """Разрешить пути к GGUF и mmproj: env → явные пути → каталог models/."""
    out_model = Path(model_path) if model_path else None
    out_mmproj = Path(mmproj_path) if mmproj_path else None
    if os.environ.get("LLAMA_MODEL_PATH"):
        out_model = Path(os.environ["LLAMA_MODEL_PATH"])
    if os.environ.get("LLAMA_MMPROJ_PATH"):
        out_mmproj = Path(os.environ["LLAMA_MMPROJ_PATH"])
    if out_model is not None and out_mmproj is not None:
        return out_model, out_mmproj
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    candidate_model = models_dir / _DEFAULT_LLAMA_MODEL
    candidate_mmproj = models_dir / _DEFAULT_LLAMA_MMPROJ
    if candidate_model.exists() and candidate_mmproj.exists():
        return candidate_model, candidate_mmproj
    return out_model, out_mmproj


def _detect_backend() -> str:
    """Detect which backend is available: transformers or llama_cpp."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        _BACKEND = "llama_cpp"
        return _BACKEND
    except ImportError:
        pass
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        _BACKEND = "transformers"
        return _BACKEND
    except ImportError:
        raise RuntimeError(
            "No backend available. Install either:\n"
            "  - transformers, torch, accelerate, qwen-vl-utils (recommended)\n"
            "  - llama-cpp-python (with Qwen2.5-VL support)"
        )


def get_backend() -> str:
    """Return the current VLM backend in use: 'llama_cpp' or 'transformers'."""
    return _detect_backend()


def _to_data_uri(p: Path) -> str:
    import base64
    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    suffix = p.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{data}"


def _extract_llama_cpp(
    image_path: Path,
    model_path: Optional[Path],
    mmproj_path: Optional[Path],
    use_gpu: bool,
    max_tokens: int,
    n_ctx: int,
) -> str:
    """Extract using cached llama-cpp-python + Qwen25VLChatHandler."""
    llm = _get_llama_cpp_vlm(use_gpu, model_path, mmproj_path, n_ctx)
    data_uri = _to_data_uri(image_path)
    messages = [
        {"role": "system", "content": "Ты извлекаешь из диаграмм список шагов по стрелкам (сверху вниз, справа налево). Если есть дорожки BPMN с ролями — выводи «Шаг | Роль» и для каждого шага «N. Текст | Роль». Конечные события (кружок «принята»/«отклонена») не выводи как шаг. Только текст из фигур; подписи на стрелках не включай. Ответ на русском."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": DIAGRAM_PROMPT},
            ],
        },
    ]
    with _VLM_LOCK:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
        )
    return response["choices"][0]["message"]["content"].strip()


def _extract_transformers(
    image_path: Path,
    use_gpu: bool,
    max_tokens: int,
) -> str:
    """Extract using cached Hugging Face Transformers + Qwen2.5-VL."""
    from qwen_vl_utils import process_vision_info

    model, processor, device = _get_transformers_vlm(use_gpu)
    image_path_str = str(image_path.resolve())
    messages = [
        {
            "role": "system",
            "content": "Ты извлекаешь из диаграмм список шагов по стрелкам (сверху вниз, справа налево). Если есть дорожки BPMN с ролями — выводи «Шаг | Роль» и для каждого шага «N. Текст | Роль». Конечные события (кружок «принята»/«отклонена») не выводи как шаг. Только текст из фигур; подписи на стрелках не включай. Ответ на русском.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path_str},
                {"type": "text", "text": DIAGRAM_PROMPT},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    with _VLM_LOCK:
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def extract_algorithm(
    image_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    mmproj_path: Optional[Union[str, Path]] = None,
    use_gpu: bool = False,
    max_tokens: int = 1024,
    n_ctx: int = 2048,
    use_preprocessing: bool = True,
) -> str:
    """
    Extract algorithm description from a diagram file or image using Qwen2.5-VL-3B.

    Supported formats: .png, .jpg, .jpeg, .gif, .webp (image); .svg (convert to image);
    .bpmn, .drawio, .xml (BPMN) — parse XML; .uml, .puml — render PlantUML then VLM.

    Args:
        image_path: Path to the diagram (image or .drawio, .bpmn, .svg, .xml, .uml).
        model_path: Path to GGUF model (llama_cpp only). If None, uses HF hub.
        mmproj_path: Path to mmproj vision encoder (llama_cpp only).
        use_gpu: If True, use GPU for inference.
        max_tokens: Maximum tokens to generate.
        n_ctx: Context window size (llama_cpp only).
        use_preprocessing: If True, apply image preprocessing (upscale/contrast) before VLM.

    Returns:
        Text description of the algorithm/process.
    """
    from src.diagram_formats import (
        SUPPORTED_EXTENSIONS,
        convert_svg_to_png,
        is_bpmn_xml,
        parse_bpmn,
        parse_drawio,
        render_plantuml_to_png,
    )
    from src.image_preprocessing import preprocess_for_vlm

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: {ext}. Supported: .png, .jpg, .svg, .bpmn, .drawio, .xml, .uml, .puml"
        )

    # Парсинг BPMN XML (без VLM)
    if ext == ".bpmn" or (ext == ".xml" and is_bpmn_xml(path)):
        result = parse_bpmn(path)
        if result:
            return result
        if ext == ".bpmn":
            raise ValueError("Failed to parse BPMN file")

    # Парсинг draw.io (без VLM)
    if ext == ".drawio":
        result = parse_drawio(path)
        if result:
            return result
        raise ValueError("Failed to parse draw.io file")

    # Конвертация в изображение и вызов VLM
    if ext == ".svg":
        png_path = convert_svg_to_png(path)
        try:
            return extract_algorithm(
                png_path,
                model_path,
                mmproj_path,
                use_gpu,
                max_tokens,
                n_ctx,
                use_preprocessing,
            )
        finally:
            png_path.unlink(missing_ok=True)

    if ext in (".uml", ".puml"):
        png_path = render_plantuml_to_png(path)
        if png_path:
            try:
                return extract_algorithm(
                    png_path,
                    model_path,
                    mmproj_path,
                    use_gpu,
                    max_tokens,
                    n_ctx,
                    use_preprocessing,
                )
            finally:
                png_path.unlink(missing_ok=True)
        raise ValueError(
            "PlantUML render failed. Install: pip install plantuml (uses online server)"
        )

    # Растровое изображение — препроцессинг и VLM
    preprocessed_path = preprocess_for_vlm(path, enabled=use_preprocessing)
    try:
        backend = _detect_backend()
        if backend == "llama_cpp":
            resolved_model, resolved_mmproj = _resolve_llama_paths(
                Path(model_path) if model_path else None,
                Path(mmproj_path) if mmproj_path else None,
            )
            return _extract_llama_cpp(
                preprocessed_path,
                resolved_model,
                resolved_mmproj,
                use_gpu,
                max_tokens,
                n_ctx,
            )
        return _extract_transformers(preprocessed_path, use_gpu, max_tokens)
    finally:
        if preprocessed_path != path and preprocessed_path.exists():
            preprocessed_path.unlink(missing_ok=True)
