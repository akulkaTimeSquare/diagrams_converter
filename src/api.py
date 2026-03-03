"""
REST API for extracting algorithms from diagrams and generating diagrams from text.

Provides FastAPI application with Swagger UI at /docs.
A single VLM is loaded at startup and reused for both extract and generate-diagram endpoints.
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


# Constants & app setup

SRC_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(SRC_DIR / "templates"))


def _parse_bool_env(name: str, default: bool = False) -> bool:
    """Parse env var as boolean (1, true, yes)"""
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def _run_warmup(use_gpu: bool, warmup_runs: int) -> None:
    """Run warmup inference to prime the VLM"""
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Preload VLM on startup so the first request does not wait

    Skips preload if SKIP_PRELOAD env is set. Optionally runs warmup inference
    when WARMUP_RUNS > 0
    """
    use_gpu = _parse_bool_env("USE_GPU")
    if _parse_bool_env("SKIP_PRELOAD"):
        logger.info("Skipping VLM preload (SKIP_PRELOAD enabled)")
        yield
        return

    try:
        ensure_vlm_loaded(use_gpu=use_gpu)
        backend = get_backend()
        logger.info("VLM preloaded successfully (backend: %s)", backend)

        warmup_runs = int(os.environ.get("WARMUP_RUNS", "1"))
        if warmup_runs > 0:
            _run_warmup(use_gpu, warmup_runs)
    except Exception as e:
        logger.warning("VLM preload failed (first request may be slow or fail): %s", e)

    yield


app = FastAPI(
    title="Diagram Algorithm Extraction API",
    description="Extract algorithm from diagrams (VLM, BPMN, drawio) and generate diagrams from text (PlantUML -> PNG). A single VLM is loaded at startup. Swagger UI provides interactive documentation.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=str(SRC_DIR / "static")),
    name="static",
)


# Pydantic models

class GenerateDiagramRequest(BaseModel):
    """Request body for generating a diagram from algorithm text"""

    algorithm_text: str = Field(..., description="Algorithm text (list of steps or process description)")
    output_format: Literal["png", "puml"] = Field(
        default="png",
        alias="format",
        description="Response format: puml — PlantUML code only; png — code + base64 image",
    )
    use_gpu: bool = Field(default=False, description="Use GPU for VLM inference")
    max_tokens: int = Field(default=1024, description="Maximum tokens for PlantUML generation")

    model_config = {"populate_by_name": True}


# Request/response helpers

async def _save_upload_to_temp(file: UploadFile, suffix: str) -> Path:
    """Save uploaded file to a temporary file. Caller must unlink when done"""
    content = await file.read()
    fd, path = tempfile.mkstemp(suffix=suffix or ".bin")
    try:
        os.write(fd, content)
        return Path(path)
    finally:
        os.close(fd)


def _load_png_and_cleanup(png_path: Path | None) -> bytes | None:
    """Read PNG bytes and delete temp file. Returns None if path is None"""
    if png_path is None:
        return None
    try:
        return png_path.read_bytes()
    finally:
        png_path.unlink(missing_ok=True)


def _format_error_with_traceback(exc: Exception) -> str:
    """Format exception with traceback for 500 response"""
    return f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"


# UI & API info

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui_home(request: Request):
    """Serve the main UI page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api")
async def api_root():
    """API root: links to documentation and main endpoints"""
    return {
        "service": "Diagram Algorithm Extraction API",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi_json": "/openapi.json",
        "health": "/health",
        "extract": "POST /extract (upload file)",
        "generate_diagram": "POST /generate-diagram (text -> diagram)",
    }


@app.get("/health")
async def health():
    """Health check. The backend field indicates which VLM is in use: llama_cpp or transformers"""
    try:
        backend = get_backend()
    except RuntimeError:
        backend = None
    return {"status": "ok", "backend": backend}


@app.get("/formats")
async def formats():
    """List of supported file extensions for diagram uploads"""
    return {
        "extensions": sorted(SUPPORTED_EXTENSIONS),
        "description": "Upload files with any of these extensions to POST /extract",
    }


# Extract

@app.post("/extract")
async def extract(
    file: Annotated[UploadFile, File(description="Diagram file: .png, .bpmn, .drawio, .svg, .xml, .uml, etc.")],
    use_gpu: Annotated[bool, Form(description="Use GPU for VLM")] = False,
    max_tokens: Annotated[int, Form(description="Max tokens for VLM response")] = 1024,
    preprocess: Annotated[bool, Form(description="Apply image preprocessing for better VLM reading")] = True,
):
    """
    Extract algorithm from an uploaded diagram

    - **Images** (.png, .jpg, .svg, etc.): sent to VLM
    - **BPMN / drawio**: XML parsing without model, fast response
    - With **preprocess=true**: apply image preprocessing before VLM
    - Returns text description of the algorithm/process
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format: {ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    try:
        tmp_path = await _save_upload_to_temp(file, ext or ".bin")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")

    try:
        request_start = time.perf_counter()
        result = extract_algorithm(
            tmp_path,
            use_gpu=use_gpu,
            max_tokens=max_tokens,
            use_preprocessing=preprocess,
        )
        elapsed = time.perf_counter() - request_start
        logger.info("extract request total=%.4fs file=%s format=%s", elapsed, file.filename, ext)

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


# Generate diagram

_GENERATE_RESPONSES = {
    200: {
        "description": "With **download=false**: JSON with plantuml and image_base64. With **download=true** and output_format=png: diagram.png file for download.",
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "plantuml": {"type": "string", "description": "PlantUML source code"},
                        "image_base64": {"type": "string", "nullable": True, "description": "PNG as base64 (when format=png)"},
                        "render_note": {"type": "string", "description": "Hint when render failed"},
                    },
                },
            },
            "image/png": {
                "schema": {"type": "string", "format": "binary"},
                "description": "diagram.png file (when download=true and output_format=png)",
            },
        },
    },
    422: {"description": "Validation error or PNG not obtained when download=true"},
    500: {"description": "Internal server error"},
}


@app.post(
    "/generate-diagram",
    tags=["Generate"],
    summary="Generate diagram from algorithm text",
    responses=_GENERATE_RESPONSES,
)
async def generate_diagram_endpoint(
    body: GenerateDiagramRequest,
    download: bool = Query(False, description="Return PNG as downloadable file (only when output_format=png)"),
):
    """
    Send text to the shared VLM which outputs PlantUML activity diagram code
    With format=png the code is rendered to PNG via PlantUML service
    """
    if download and body.output_format != "png":
        raise HTTPException(
            status_code=422,
            detail="PNG download requires output_format=png. Set format=png in the request body.",
        )

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
        raise HTTPException(status_code=500, detail=_format_error_with_traceback(e))

    png_bytes = _load_png_and_cleanup(png_path)

    if download:
        if png_bytes is None:
            snippet = plantuml_source[:500] + ("..." if len(plantuml_source) > 500 else "")
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "PNG rendering failed. Check server logs. Test PlantUML at https://www.plantuml.com/plantuml/uml/",
                    "plantuml_snippet": snippet,
                },
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
            result["render_note"] = "PlantUML render failed (check syntax or install plantuml)."
    return result
