"""
REST API для извлечения алгоритма из диаграмм и генерации диаграмм из текста.
FastAPI + Swagger UI на /docs.
Одна VLM загружается при старте (lifespan) и переиспользуется для extract и generate-diagram.
"""
import base64
import logging
import os
import tempfile
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.diagram_extractor import ensure_vlm_loaded, extract_algorithm, get_backend
from src.diagram_formats import SUPPORTED_EXTENSIONS
from src.diagram_generator import generate_diagram

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload VLM on startup so the first request does not wait."""
    use_gpu = os.environ.get("USE_GPU", "").lower() in ("1", "true", "yes")
    if os.environ.get("SKIP_PRELOAD", "").lower() in ("1", "true", "yes"):
        logger.info("Skipping VLM preload (SKIP_PRELOAD enabled)")
        yield
        return
    try:
        ensure_vlm_loaded(use_gpu=use_gpu)
        backend = get_backend()
        logger.info("VLM preloaded successfully (backend: %s)", backend)
        warmup_runs = int(os.environ.get("WARMUP_RUNS", "1"))
        if warmup_runs > 0:
            from PIL import Image
            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            try:
                Image.new("RGB", (16, 16), color=(255, 255, 255)).save(tmp_path)
                for i in range(warmup_runs):
                    try:
                        extract_algorithm(
                            tmp_path,
                            use_gpu=use_gpu,
                            max_tokens=16,
                            use_preprocessing=False,
                            log_timings=False,
                        )
                        logger.info("Warmup run %d/%d completed", i + 1, warmup_runs)
                    except Exception as e:
                        logger.warning("Warmup run %d/%d failed: %s", i + 1, warmup_runs, e)
                        break
            finally:
                Path(tmp_path).unlink(missing_ok=True)
    except Exception as e:
        logger.warning("VLM preload failed (first request may be slow or fail): %s", e)
    yield
    # Shutdown: nothing to tear down (model stays in process)


app = FastAPI(
    title="Diagram Algorithm Extraction API",
    description="Извлечение алгоритма из диаграмм (VLM, BPMN, drawio) и генерация диаграмм из текста алгоритма (PlantUML → PNG). Одна VLM загружается при старте. Swagger UI — интерактивная документация.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



class GenerateDiagramRequest(BaseModel):
    """Тело запроса для генерации диаграммы из текста алгоритма."""

    algorithm_text: str = Field(..., description="Текст алгоритма (список шагов или описание процесса)")
    output_format: Literal["png", "puml"] = Field(
        default="png",
        alias="format",
        description="В ответе: puml — только код PlantUML; png — код + изображение (base64)",
    )
    use_gpu: bool = Field(default=False, description="Использовать GPU для генерации")
    max_tokens: int = Field(default=1024, description="Максимум токенов для генерации PlantUML")

    model_config = {"populate_by_name": True}


@app.get("/api")
async def api_root():
    """Корень API: ссылки на документацию."""
    return {
        "service": "Diagram Algorithm Extraction API",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi_json": "/openapi.json",
        "health": "/health",
        "extract": "POST /extract (upload file)",
        "generate_diagram": "POST /generate-diagram (text → diagram)",
    }


@app.get("/health")
async def health():
    """Проверка доступности сервиса. Поле backend показывает, какая VLM используется: llama_cpp или transformers."""
    try:
        backend = get_backend()
    except RuntimeError:
        backend = None
    return {"status": "ok", "backend": backend}


@app.get("/formats")
async def formats():
    """Список поддерживаемых расширений файлов."""
    return {
        "extensions": sorted(SUPPORTED_EXTENSIONS),
        "description": "Загружайте файлы с любым из этих расширений в POST /extract",
    }


@app.post("/extract")
async def extract(
    file: Annotated[UploadFile, File(description="Файл диаграммы: .png, .bpmn, .drawio, .svg, .xml, .uml и др.")],
    use_gpu: Annotated[bool, Form(description="Использовать GPU для VLM (только для изображений)")] = False,
    max_tokens: Annotated[int, Form(description="Максимум токенов ответа (для VLM)")] = 1024,
    preprocess: Annotated[bool, Form(description="Применить препроцессинг изображения для лучшего чтения VLM")] = True,
):
    """
    Извлечь алгоритм из загруженной диаграммы.

    - **Изображения** (.png, .jpg, .svg и т.д.): отправляются в VLM (Qwen2.5-VL).
    - **BPMN / drawio**: парсинг XML без модели, быстрый ответ.
    - При **preprocess=true** (по умолчанию): авто-препроцессинг (апскейл мелких, контраст) для изображений.
    - Результат — текст в формате «Шаг | Роль» или список шагов.
    """
    request_start = time.perf_counter()
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Неподдерживаемый формат: {ext}. Поддерживаются: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    try:
        suffix = ext or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения файла: {e}")

    try:
        result = extract_algorithm(
            tmp_path,
            use_gpu=use_gpu,
            max_tokens=max_tokens,
            use_preprocessing=preprocess,
        )
        total = time.perf_counter() - request_start
        logger.info("extract request total=%.4fs file=%s format=%s", total, file.filename, ext)
        return {
            "algorithm": result,
            "filename": file.filename or "upload",
            "format": ext,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post(
    "/generate-diagram",
    tags=["Generate"],
    summary="Сгенерировать диаграмму по тексту алгоритма",
    responses={
        200: {
            "description": "При **download=false**: JSON с полями plantuml и image_base64. При **download=true** и output_format=png: файл diagram.png для скачивания.",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "plantuml": {"type": "string", "description": "Код PlantUML"},
                            "image_base64": {"type": "string", "nullable": True, "description": "PNG в base64 (при format=png)"},
                            "render_note": {"type": "string", "description": "Подсказка при неудачном рендере"},
                        },
                    },
                },
                "image/png": {
                    "schema": {"type": "string", "format": "binary"},
                    "description": "Файл diagram.png (при download=true и output_format=png)",
                },
            },
        },
        422: {"description": "Ошибка валидации или PNG не получен при download=true"},
        500: {"description": "Внутренняя ошибка сервера"},
    },
)
async def generate_diagram_endpoint(
    body: GenerateDiagramRequest,
    download: bool = Query(False, description="Вернуть PNG файлом для скачивания (только при output_format=png)"),
):
    """
    Текст отправляется в общую VLM (Qwen2.5-VL), которая выдаёт код PlantUML
    activity-диаграммы. При format=png код рендерится в PNG (через сервис PlantUML).
    В Swagger: включите **download** и выберите **output_format**: `png` — в ответ придёт файл diagram.png.
    """
    try:
        plantuml_source, png_path = generate_diagram(
            body.algorithm_text,
            output_format=body.output_format,
            use_gpu=body.use_gpu,
            max_tokens=body.max_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("generate-diagram failed")
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n\n{tb}")

    png_bytes = None
    if body.output_format == "png" and png_path is not None:
        try:
            png_bytes = png_path.read_bytes()
        finally:
            png_path.unlink(missing_ok=True)

    if download:
        if png_bytes is None:
            raise HTTPException(
                status_code=422,
                detail="PNG не получен (рендер не удался). Укажите output_format=png.",
            )
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=\"diagram.png\""},
        )

    result: dict = {"plantuml": plantuml_source}
    if body.output_format == "png":
        result["image_base64"] = base64.b64encode(png_bytes).decode("utf-8") if png_bytes else None
        if not png_bytes:
            result["render_note"] = "Рендер PlantUML не удался (проверьте синтаксис или установите plantuml)."
    return result
