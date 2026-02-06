"""
Extract algorithm/process description from diagram images using Qwen2.5-VL-3B.
Supports: Transformers (primary), llama-cpp-python (optional, when available).
Single VLM instance is cached and reused for both extract (image→text) and generate (text→PlantUML).
"""
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional, Union

# Serialize access to the single VLM instance (no concurrent inference).
_VLM_LOCK = threading.Lock()

# Пути по умолчанию для llama.cpp (каталог проекта/models/)
# По умолчанию q8_0; q4_k_m — экономия RAM
_LLAMA_QUANT_OPTIONS = (
    "q4_0", "q4_k_s", "q4_k_m", "q5_k_m", "q8_0",
    "f16-q8_0", "bf16-q8_0", "bf16",  # bf16 — полная точность без квантизации
)
_DEFAULT_LLAMA_MMPROJ = "Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf"


def _get_llama_quant() -> str:
    """Квантизация GGUF из env LLAMA_QUANT (по умолчанию q8_0)."""
    q = os.environ.get("LLAMA_QUANT", "q8_0").strip()
    if q in _LLAMA_QUANT_OPTIONS:
        return q
    if q.lower() in ("f16-q8_0", "bf16-q8_0", "bf16"):
        if q.lower() == "bf16":
            return "bf16"
        return "f16-q8_0" if "f16" in q.lower() else "bf16-q8_0"
    return "q8_0"


def _default_llama_model_filename() -> str:
    return f"Qwen2.5-VL-3B-Instruct-{_get_llama_quant()}.gguf"

# Промпт для извлечения алгоритма в формате, совпадающем с test.txt
DIAGRAM_PROMPT = """Обязательно: всегда выдавай ответ. Пустой вывод не допускается.

Задача (как в README): извлечь из диаграммы текст алгоритма/процесса. Ответ — только на русском.

Верни ТОЛЬКО один из двух форматов (без markdown, без пояснений, без вступлений):

1) Если на диаграмме есть дорожки (swimlanes) с подписями ролей:
Шаг | Роль
1. <дословная подпись внутри фигуры> | <дословная подпись дорожки>
2. ...

2) Если дорожек нет:
Шаг
1. <дословная подпись внутри фигуры>
2. ...

Правила (анти-галлюцинации):
- Пиши только то, что НАПИСАНО на диаграмме (в фигурах и на дорожках). Никаких фактов/шагов/ролей “по смыслу”.
- Текст шага — дословно из подписи фигуры (прямоугольник/ромб/событие с подписью). Не используй шаблоны/скобки вроде [Инициализация], [Выбор], и не перефразируй.
- Роль у шага — только подпись дорожки. Если дорожек нет — колонку «Роль» не выводи вообще.
- Формат должен быть консистентным: если заголовок «Шаг | Роль» — то КАЖДАЯ строка шага обязана содержать « | <роль>». Если дорожек нет — никакого символа «|» в ответе быть не должно.
- Стрелки/связи: перечисляй узлы по направлению процесса (как идут стрелки). При ветвлении: сначала одна ветка до конца, затем другая, затем общее продолжение.
- Конечные события без подписи не выводи как шаг.
- Если подпись не читается — напиши НЕРАЗБОРЧИВО (не придумывай).

Если на изображении нет пошагового процесса (не BPMN/flowchart) — тогда вместо форматов выше:
Описание
<кратко перечисли, что видно на изображении, без выдуманных деталей>"""

# Резервный сверх-строгий промпт: используется при нарушении формата/сильной галлюцинации.
STRICT_DIAGRAM_PROMPT = """Верни ответ строго в одном из форматов ниже. Никаких других заголовков/фраз.

Формат A (есть swimlanes):
Шаг | Роль
1. <подпись фигуры> | <подпись дорожки>
2. ...

Формат B (swimlanes нет):
Шаг
1. <подпись фигуры>
2. ...

Формат C (это не процесс):
Описание
<кратко что видно>

Самопроверка перед ответом:
- Первая строка должна быть РОВНО: «Шаг», «Шаг | Роль» или «Описание».
- Если выбрал «Шаг | Роль» — в каждой строке шагов должен быть символ «|» и роль.
- Если текста не видно — «НЕРАЗБОРЧИВО» (не придумывай)."""

_BACKEND: Optional[str] = None

logger = logging.getLogger(__name__)

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
        device_map_env = os.environ.get("FORCE_DEVICE_MAP")
        torch_dtype = "float16" if use_gpu else "auto"
        load_in_8bit = os.environ.get("LOAD_IN_8BIT", "").lower() in ("1", "true", "yes")
        load_in_4bit = os.environ.get("LOAD_IN_4BIT", "").lower() in ("1", "true", "yes")
        device_map = device_map_env if device_map_env else ("auto" if use_gpu else None)

        if use_gpu:
            try:
                cuda_ver = getattr(__import__("torch").version, "cuda", None)
                cuda_dev = __import__("torch").cuda.get_device_name(0)
                logger.info("cuda_available=%s torch_cuda=%s", True, cuda_ver)
                logger.info("cuda_device=%s", cuda_dev)
                print(f"cuda_available=True torch_cuda={cuda_ver}")
                print(f"cuda_device={cuda_dev}")
                try:
                    free_mem, total_mem = __import__("torch").cuda.mem_get_info()
                    logger.info("vram_before free=%d total=%d", free_mem, total_mem)
                    print(f"vram_before free={free_mem} total={total_mem}")
                except Exception as e:
                    logger.info("vram_before unavailable: %s", e)
                    print(f"vram_before unavailable: {e}")
            except Exception as e:
                logger.info("cuda_info unavailable: %s", e)
                print(f"cuda_info unavailable: {e}")

        logger.info(
            "transformers_load config: device_map=%s dtype=%s load_in_8bit=%s load_in_4bit=%s",
            device_map,
            torch_dtype,
            load_in_8bit,
            load_in_4bit,
        )
        print(
            "transformers_load config: "
            f"device_map={device_map} dtype={torch_dtype} load_in_8bit={load_in_8bit} load_in_4bit={load_in_4bit}"
        )
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
        }
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        _transformers_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs,
        )
        if not use_gpu:
            _transformers_model = _transformers_model.to(device)
        if use_gpu:
            try:
                param = next(_transformers_model.parameters())
                logger.info("model_param device=%s dtype=%s", param.device, param.dtype)
                print(f"model_param device={param.device} dtype={param.dtype}")
            except Exception as e:
                logger.info("model_param unavailable: %s", e)
                print(f"model_param unavailable: {e}")
            try:
                free_mem, total_mem = __import__("torch").cuda.mem_get_info()
                logger.info("vram_after free=%d total=%d", free_mem, total_mem)
                print(f"vram_after free={free_mem} total={total_mem}")
            except Exception as e:
                logger.info("vram_after unavailable: %s", e)
                print(f"vram_after unavailable: {e}")
        _transformers_model.eval()
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
            quant = _get_llama_quant()
            # Точное имя файла, чтобы не совпало с bf16-q8_0 / f16-q8_0 и т.д.
            llm_filename = f"Qwen2.5-VL-3B-Instruct-{quant}.gguf"
            _llama_chat_handler = Qwen25VLChatHandler.from_pretrained(
                repo_id="Mungert/Qwen2.5-VL-3B-Instruct-GGUF",
                filename="Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf",
            )
            _llama_llm = Llama.from_pretrained(
                repo_id="Mungert/Qwen2.5-VL-3B-Instruct-GGUF",
                filename=llm_filename,
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
            import torch
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    repetition_penalty=1.15,
                )
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
            repeat_penalty=1.15,
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
    candidate_model = models_dir / _default_llama_model_filename()
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
    diagram_prompt: str = DIAGRAM_PROMPT,
) -> str:
    """Extract using cached llama-cpp-python + Qwen25VLChatHandler."""
    llm = _get_llama_cpp_vlm(use_gpu, model_path, mmproj_path, n_ctx)
    data_uri = _to_data_uri(image_path)
    messages = [
        {
            "role": "system",
            "content": "Ты извлекаешь из диаграмм ТОЛЬКО текст алгоритма/процесса. Строго следуй формату из инструкции пользователя. Запрещено придумывать шаги/роли/факты, писать вступления/описания перед списком, или заменять подписи шаблонами. Если текст не читается — пиши «НЕРАЗБОРЧИВО». Ответ на русском. Пустой ответ запрещён.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": diagram_prompt},
            ],
        },
    ]
    with _VLM_LOCK:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            repeat_penalty=1.1,
        )
    return response["choices"][0]["message"]["content"].strip()


def _extract_transformers(
    image_path: Path,
    use_gpu: bool,
    max_tokens: int,
    diagram_prompt: str = DIAGRAM_PROMPT,
) -> str:
    """Extract using cached Hugging Face Transformers + Qwen2.5-VL."""
    from qwen_vl_utils import process_vision_info

    model, processor, device = _get_transformers_vlm(use_gpu)
    image_path_str = str(image_path.resolve())
    messages = [
        {
            "role": "system",
            "content": "Ты извлекаешь из диаграмм ТОЛЬКО текст алгоритма/процесса. Строго следуй формату из инструкции пользователя. Запрещено придумывать шаги/роли/факты, писать вступления/описания перед списком, или заменять подписи шаблонами. Если текст не читается — пиши «НЕРАЗБОРЧИВО». Ответ на русском. Пустой ответ запрещён.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path_str},
                {"type": "text", "text": diagram_prompt},
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
        import torch
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                repetition_penalty=1.15,
            )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def _is_hallucinated_generic(text: str) -> bool:
    """Detect generic BPMN hallucination (Создание заявки, Проверка бюджета, Утверждение)."""
    t = text.lower()
    return (
        "создание заявки" in t
        and "проверка бюджета" in t
        and "утверждение" in t
        and t.count("|") >= 3  # role column present
    )


def _strip_description_intro(text: str) -> str:
    """Убрать ведущий абзац «Описание: ...» / «Диаграмма BPMN ...», если дальше идёт список шагов."""
    if not text or "шаг" not in text.lower():
        return text
    lines = text.split("\n")
    start_idx = 0
    for i, line in enumerate(lines):
        s = line.strip().lower()
        if s.startswith("шаг") or (s and s[0].isdigit() and ("." in s[:4] or ")" in s[:4])):
            start_idx = i
            break
        if s.startswith("описание") or "диаграмма bpmn" in s or "процессный бизнес" in s:
            continue
        if s and not s.startswith("описание"):
            start_idx = i
            break
    if start_idx > 0:
        return "\n".join(lines[start_idx:]).strip()
    return text


def _normalize_extracted_text(text: str) -> str:
    """
    Normalize common VLM output glitches while preserving content:
    - Drop empty line after header
    - Enforce header/column consistency ("Шаг" vs "Шаг | Роль")
    - If header has role column but no step lines have roles -> downgrade to "Шаг"
    - If header is "Шаг" but steps contain " | " -> keep only step text (left side)
    """
    if not text:
        return text
    raw_lines = [ln.rstrip() for ln in text.strip().split("\n")]
    if not raw_lines:
        return text

    # Preserve "Описание" mode as-is (only strip trailing spaces)
    first_nonempty_idx = next((i for i, ln in enumerate(raw_lines) if ln.strip()), None)
    if first_nonempty_idx is None:
        return text.strip()
    first = raw_lines[first_nonempty_idx].strip()
    if first.lower().startswith("описание"):
        return "\n".join(raw_lines).strip()

    # Find header line (expect "Шаг" or "Шаг | Роль")
    header_idx = first_nonempty_idx
    header = raw_lines[header_idx].strip()
    header_low = header.lower().replace("\t", " ")
    if not header_low.startswith("шаг"):
        # Unknown header; still remove empty lines at the top
        return "\n".join(raw_lines[header_idx:]).strip()

    # Remove single empty line right after header
    lines = raw_lines[:]
    if header_idx + 1 < len(lines) and not lines[header_idx + 1].strip():
        lines.pop(header_idx + 1)

    header = lines[header_idx].strip()
    has_role_header = "|" in header and "роль" in header.lower()

    # Collect step lines (after header)
    step_lines = lines[header_idx + 1 :]
    step_lines_stripped = [ln.strip() for ln in step_lines if ln.strip()]
    step_lines_have_roles = any(" | " in ln for ln in step_lines_stripped)

    if has_role_header and not step_lines_have_roles:
        # Header says roles, but body doesn't -> downgrade header.
        lines[header_idx] = "Шаг"
        return "\n".join(lines[header_idx:]).strip()

    if (not has_role_header) and step_lines_have_roles:
        # Body has roles, but header doesn't -> drop role column to keep deterministic format.
        new_step_lines = []
        for ln in step_lines:
            if " | " in ln:
                new_step_lines.append(ln.split(" | ", 1)[0].rstrip())
            else:
                new_step_lines.append(ln)
        lines = lines[: header_idx + 1] + new_step_lines
        lines[header_idx] = "Шаг"
        return "\n".join(lines[header_idx:]).strip()

    return "\n".join(lines[header_idx:]).strip()


def _is_invalid_extracted_format(text: str) -> bool:
    """Return True if output violates required header/column rules."""
    if not text:
        return True
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return True
    header = lines[0]
    h = header.lower()
    if h == "описание":
        return False
    if not h.startswith("шаг"):
        return True

    has_role_header = ("|" in header) and ("роль" in h)
    step_lines = lines[1:]
    numbered = [ln for ln in step_lines if ln and ln[0].isdigit()]
    if not numbered:
        # empty list isn't useful (but prompt says never empty) -> treat as invalid to trigger retry
        return True

    if has_role_header:
        # every numbered line must have a role column
        return any(" | " not in ln for ln in numbered)

    # no role header: must not contain role column in steps
    return any(" | " in ln for ln in numbered)


def extract_algorithm(
    image_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    mmproj_path: Optional[Union[str, Path]] = None,
    use_gpu: bool = False,
    max_tokens: int = 1024,
    n_ctx: int = 2048,
    use_preprocessing: bool = True,
    log_timings: bool = True,
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
    total_start = time.perf_counter()
    preprocess_time = 0.0
    inference_time = 0.0
    postprocess_time = 0.0
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: {ext}. Supported: .png, .jpg, .svg, .bpmn, .drawio, .xml, .uml, .puml"
        )

    # Парсинг BPMN XML (без VLM)
    if ext == ".bpmn" or (ext == ".xml" and is_bpmn_xml(path)):
        parse_start = time.perf_counter()
        result = parse_bpmn(path)
        preprocess_time += time.perf_counter() - parse_start
        if result:
            if log_timings:
                total = time.perf_counter() - total_start
                print(
                    f"timings extract (bpmn): preprocess={preprocess_time:.4f}s "
                    f"inference={inference_time:.4f}s postprocess={postprocess_time:.4f}s total={total:.4f}s"
                )
                logger.info(
                    "timings extract (bpmn): preprocess=%.4fs inference=%.4fs postprocess=%.4fs total=%.4fs",
                    preprocess_time,
                    inference_time,
                    postprocess_time,
                    total,
                )
            return result
        if ext == ".bpmn":
            raise ValueError("Failed to parse BPMN file")

    # Парсинг draw.io (без VLM)
    if ext == ".drawio":
        parse_start = time.perf_counter()
        result = parse_drawio(path)
        preprocess_time += time.perf_counter() - parse_start
        if result:
            if log_timings:
                total = time.perf_counter() - total_start
                print(
                    f"timings extract (drawio): preprocess={preprocess_time:.4f}s "
                    f"inference={inference_time:.4f}s postprocess={postprocess_time:.4f}s total={total:.4f}s"
                )
                logger.info(
                    "timings extract (drawio): preprocess=%.4fs inference=%.4fs postprocess=%.4fs total=%.4fs",
                    preprocess_time,
                    inference_time,
                    postprocess_time,
                    total,
                )
            return result
        raise ValueError("Failed to parse draw.io file")

    # Конвертация в изображение и вызов VLM
    if ext == ".svg":
        convert_start = time.perf_counter()
        png_path = convert_svg_to_png(path)
        preprocess_time += time.perf_counter() - convert_start
        try:
            return extract_algorithm(
                png_path,
                model_path,
                mmproj_path,
                use_gpu,
                max_tokens,
                n_ctx,
                use_preprocessing,
                log_timings=False,
            )
        finally:
            png_path.unlink(missing_ok=True)

    if ext in (".uml", ".puml"):
        render_start = time.perf_counter()
        png_path = render_plantuml_to_png(path)
        preprocess_time += time.perf_counter() - render_start
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
                    log_timings=False,
                )
            finally:
                png_path.unlink(missing_ok=True)
        raise ValueError(
            "PlantUML render failed. Install: pip install plantuml (uses online server)"
        )

    # Растровое изображение — препроцессинг и VLM
    def _run_vlm(preproc: bool, diagram_prompt: str) -> str:
        pre_start = time.perf_counter()
        p = preprocess_for_vlm(path, enabled=preproc)
        nonlocal preprocess_time
        preprocess_time += time.perf_counter() - pre_start
        try:
            backend = _detect_backend()
            if backend == "llama_cpp":
                resolved_model, resolved_mmproj = _resolve_llama_paths(
                    Path(model_path) if model_path else None,
                    Path(mmproj_path) if mmproj_path else None,
                )
                infer_start = time.perf_counter()
                r = _extract_llama_cpp(
                    p,
                    resolved_model,
                    resolved_mmproj,
                    use_gpu,
                    max_tokens,
                    n_ctx,
                    diagram_prompt=diagram_prompt,
                )
                nonlocal inference_time
                inference_time += time.perf_counter() - infer_start
            else:
                infer_start = time.perf_counter()
                r = _extract_transformers(p, use_gpu, max_tokens, diagram_prompt=diagram_prompt)
                inference_time += time.perf_counter() - infer_start
            return r
        finally:
            if p != path and p.exists():
                p.unlink(missing_ok=True)

    try:
        result = _run_vlm(use_preprocessing, DIAGRAM_PROMPT)
        if use_preprocessing and _is_hallucinated_generic(result):
            logger.info("Retrying extract without preprocessing (hallucination detected)")
            result = _run_vlm(False, DIAGRAM_PROMPT)

        normalized = _normalize_extracted_text(_strip_description_intro(result))
        if _is_invalid_extracted_format(normalized):
            logger.info("Retrying extract with STRICT_DIAGRAM_PROMPT (format violation)")
            result2 = _run_vlm(use_preprocessing, STRICT_DIAGRAM_PROMPT)
            if use_preprocessing and _is_hallucinated_generic(result2):
                logger.info("Retrying strict extract without preprocessing (hallucination detected)")
                result2 = _run_vlm(False, STRICT_DIAGRAM_PROMPT)
            normalized2 = _normalize_extracted_text(_strip_description_intro(result2))
            if not _is_invalid_extracted_format(normalized2):
                normalized = normalized2

        return normalized
    finally:
        if log_timings:
            total = time.perf_counter() - total_start
            print(
                f"timings extract: preprocess={preprocess_time:.4f}s inference={inference_time:.4f}s "
                f"postprocess={postprocess_time:.4f}s total={total:.4f}s"
            )
            logger.info(
                "timings extract: preprocess=%.4fs inference=%.4fs postprocess=%.4fs total=%.4fs",
                preprocess_time,
                inference_time,
                postprocess_time,
                total,
            )
