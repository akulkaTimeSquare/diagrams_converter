# Diagram-to-Algorithm Extraction Service

Извлечение алгоритма из диаграмм (PNG, BPMN, drawio) и генерация PlantUML из текста. VLM: Qwen2.5-VL-3B.

## Структура проекта

```
├── src/
│   ├── api.py                  # FastAPI
│   ├── diagram_extractor.py    # Извлечение
│   ├── diagram_generator.py    # Генерация
│   ├── diagram_formats.py      # Поддержка форматов
│   ├── diagram_prompts.py
│   ├── image_preprocessing.py
│   ├── templates/              # Веб-интерфейс
│   └── static/
├── scripts/
│   └── download_models.py      # Скачивание GGUF для llama-cpp
├── tests/
│   ├── test_formats.py
│   └── test_qwen_integration.py
├── Dockerfile                  # llama-cpp + CUDA
├── docker-compose.yml
├── requirements.txt            # Transformers
└── requirements-llamacpp.txt
```

## Установка

Transformers:
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

llama-cpp:
```powershell
pip install -r requirements-llamacpp.txt
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
python scripts/download_models.py
```

## Запуск

Локально:

```powershell
$env:USE_GPU = "true"
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Docker (llama-cpp + GPU):

```powershell
docker compose up -d --build
```

## URL

| URL | Описание |
|-----|----------|
| http://localhost:8000/ | UI |
| http://localhost:8000/docs | Swagger UI |
| http://localhost:8000/health | Статус и бэкенд |
| http://localhost:8000/formats | Поддерживаемые форматы |

## API

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/extract` | Файл диаграммы -> текст алгоритма |
| POST | `/generate-diagram` | Текст -> PlantUML (+ PNG) |

**POST /extract**: `file`, `use_gpu`, `max_tokens`, `preprocess`

**POST /generate-diagram** (JSON): `algorithm_text`, `format` (png/puml), `use_gpu`, `max_tokens`, `download` для PNG

## Тесты

```powershell
python tests/test_formats.py
python tests/test_qwen_integration.py
```

## Переменные окружения

| Переменная | Описание |
|------------|----------|
| USE_GPU | `true` — GPU |
| SKIP_PRELOAD | `1` — не загружать модель при старте |
| MAX_IMAGE_SIDE | ограничение размера изображения |
