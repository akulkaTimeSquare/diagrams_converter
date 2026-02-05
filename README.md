# Diagram-to-Algorithm Extraction Service

Сервис для **извлечения алгоритма из диаграмм** (изображения, BPMN, drawio) и **генерации диаграмм из текста** алгоритма. Использует одну модель **Qwen2.5-VL-3B** (VLM): для распознавания диаграмм по картинке и для генерации кода PlantUML по тексту. Доступен как Python-библиотека и как **REST API** (FastAPI + Swagger UI).

---

## Содержание

- [Описание сервиса](#описание-сервиса)
- [Требования](#требования)
- [Установка](#установка)
- [Запуск сервиса](#запуск-сервиса)
- [REST API: как обращаться](#rest-api-как-обращаться)
- [Использование из Python](#использование-из-python)
- [Поддерживаемые форматы](#поддерживаемые-форматы)
- [Бэкенды (Transformers / llama.cpp)](#бэкенды-transformers--llamacpp)
- [Структура проекта](#структура-проекта)
- [Устранение неполадок](#устранение-неполадок)
- [Лицензии](#лицензии)

---

## Описание сервиса

Сервис решает две задачи:

1. **Диаграмма → текст алгоритма**  
   Загружаете файл диаграммы (PNG, BPMN, drawio и др.) — в ответ получаете текстовое описание алгоритма или бизнес-процесса в формате «Шаг | Роль» или списка шагов. Для изображений используется VLM (Qwen2.5-VL); для BPMN/drawio — парсинг XML без модели.

2. **Текст алгоритма → диаграмма**  
   Передаёте текст с шагами алгоритма — сервис генерирует код PlantUML (activity-диаграмма) и при необходимости рендерит его в PNG (через публичный сервис PlantUML).

**Одна VLM** загружается при старте приложения и переиспользуется для обеих задач (извлечение и генерация), что экономит память и время.

---

## Требования

- **Python 3.11+** ([python.org](https://www.python.org/downloads/))
- **Память:** ~8 GB RAM на CPU или ~6 GB VRAM на GPU для модели 3B
- Для рендера PlantUML в PNG нужен доступ в интернет (используется `https://www.plantuml.com/plantuml`)

### Если Windows пишет «Python was not found»

1. Установите Python с [python.org](https://www.python.org/downloads/) и при установке включите **«Add Python to PATH»**.
2. Либо отключите алиасы Microsoft Store: **Параметры → Приложения → Дополнительные параметры приложения → Псевдонимы выполнения приложений** — выключите **python.exe** и **python3.exe**.
3. Либо используйте `py` вместо `python`: создайте venv через **`py -m venv .venv`** и запускайте скрипты через **`.venv\Scripts\python.exe`**.

---

## Установка

Из корня проекта (`diagrams`). Если команда `python` не найдена — используйте `py`:

```powershell
# Создание виртуального окружения
py -m venv .venv
# или:  python -m venv .venv

# Активация (PowerShell)
.venv\Scripts\Activate.ps1

# Установка зависимостей
.venv\Scripts\pip install -r requirements.txt
```

При первом запросе к VLM (или при старте API с предзагрузкой) модель **Qwen2.5-VL-3B** (~6 GB) будет загружена с Hugging Face. Обработка одного изображения: примерно 15–60 сек на CPU, 5–15 сек на GPU.

---

## Запуск сервиса

### REST API (FastAPI)

**С активированным venv:**

```powershell
.venv\Scripts\Activate.ps1
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

**Без активации venv (из корня проекта):**

```powershell
.venv\Scripts\uvicorn.exe src.api:app --reload --host 0.0.0.0 --port 8000
```

**С использованием GPU** (модель загружается на видеокарту):

```powershell
$env:USE_GPU = "true"
.venv\Scripts\uvicorn.exe src.api:app --reload --host 0.0.0.0 --port 8000
```

При старте сервис пытается предзагрузить VLM (lifespan). Если предзагрузка не удалась (нет памяти и т.п.), сервер всё равно запустится; первый запрос к VLM может быть медленным или завершиться ошибкой.

После запуска:

| URL | Описание |
|-----|----------|
| [http://127.0.0.1:8000/](http://127.0.0.1:8000/) | Корень API, ссылки на документацию |
| [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) | **Swagger UI** — интерактивная документация и тестирование |
| [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) | ReDoc |
| [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) | Проверка доступности сервиса |

---

## REST API: как обращаться

### Общая информация

- **Базовый URL:** `http://127.0.0.1:8000` (или ваш хост/порт).
- **Документация OpenAPI:** `GET /openapi.json`.
- Все примеры ниже предполагают, что сервер уже запущен.

### Эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/` | Информация об API и ссылки |
| GET | `/health` | Проверка доступности |
| GET | `/formats` | Список поддерживаемых расширений файлов |
| POST | `/extract` | Загрузка файла диаграммы → текст алгоритма |
| POST | `/generate-diagram` | Текст алгоритма → PlantUML и опционально PNG |

---

### POST /extract — извлечение алгоритма из диаграммы

**Параметры (form-data):**

- **file** (обязательный) — файл диаграммы (`.png`, `.jpg`, `.bpmn`, `.drawio`, `.svg`, `.xml`, `.uml`, `.puml` и др.).
- **use_gpu** (bool, по умолчанию `false`) — использовать GPU для VLM (только для изображений).
- **max_tokens** (int, по умолчанию `1024`) — максимум токенов ответа VLM.
- **preprocess** (bool, по умолчанию `true`) — применить препроцессинг изображения (апскейл мелких, улучшение контраста) для лучшего чтения VLM.

**Ответ (200):**

```json
{
  "algorithm": "Шаг\t| Роль\n1. Действие\n2. Проверка\n...",
  "filename": "diagram.png",
  "format": ".png"
}
```

**Примеры вызова:**

**PowerShell:**

```powershell
# Изображение
Invoke-RestMethod -Uri "http://127.0.0.1:8000/extract" -Method Post -Form @{
  file = Get-Item "Picture\1.png"
  preprocess = $true
}

# BPMN (без VLM)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/extract" -Method Post -Form @{
  file = Get-Item "Диаграммы\diagram.bpmn"
}
```

**curl:**

```bash
curl -X POST "http://127.0.0.1:8000/extract" \
  -F "file=@Picture/1.png" \
  -F "preprocess=true"
```

---

### POST /generate-diagram — генерация диаграммы из текста

**Тело (JSON):**

- **algorithm_text** (обязательный) — текст алгоритма (список шагов или описание процесса).
- **format** (опционально) — `"png"` или `"puml"`. По умолчанию `"png"` (код PlantUML + изображение в base64).
- **use_gpu** (bool, по умолчанию `false`) — использовать GPU для генерации.
- **max_tokens** (int, по умолчанию `1024`) — максимум токенов для генерации PlantUML.

**Query-параметр:**

- **download** (bool, по умолчанию `false`) — при `format=png` вернуть ответ как файл `diagram.png` для скачивания (вместо JSON).

**Ответ (200) при `format=png` и без `download`:**

```json
{
  "plantuml": "@startuml\nstart\n:Шаг один;\n:Шаг два;\nstop\n@enduml",
  "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "render_note": null
}
```

При неудачном рендере PNG: `image_base64` может быть `null`, в `render_note` — подсказка.

**Примеры вызова:**

**PowerShell (JSON):**

```powershell
$body = @{
  algorithm_text = "1. Начало`n2. Проверить условие`n3. Если да — действие A`n4. Если нет — действие B`n5. Конец"
  format = "png"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/generate-diagram" -Method Post -Body $body -ContentType "application/json"
```

**Скачать PNG файлом:**

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8000/generate-diagram?download=true" -Method Post `
  -Body ($body) -ContentType "application/json" -OutFile "diagram.png"
```

**curl:**

```bash
curl -X POST "http://127.0.0.1:8000/generate-diagram" \
  -H "Content-Type: application/json" \
  -d '{"algorithm_text": "1. Начало\n2. Шаг два\n3. Конец", "format": "png"}'
```

**Скачать PNG:**

```bash
curl -X POST "http://127.0.0.1:8000/generate-diagram?download=true" \
  -H "Content-Type: application/json" \
  -d '{"algorithm_text": "1. Начало\n2. Конец", "format": "png"}' \
  -o diagram.png
```

---

## Использование из Python

### Извлечение алгоритма из файла

```python
from src.diagram_extractor import extract_algorithm

# Изображение — обрабатывается VLM
result = extract_algorithm("Picture/1.png", use_gpu=False, max_tokens=256)
print(result)

# BPMN / drawio — парсинг XML, без модели
result = extract_algorithm("Диаграммы/diagram.bpmn")
result = extract_algorithm("scheme.drawio")
```

Запуск одной команды из корня проекта:

```powershell
.venv\Scripts\python.exe -c "from src.diagram_extractor import extract_algorithm; print(extract_algorithm('Picture/1.png', max_tokens=128))"
```

### Генерация диаграммы из текста

```python
from src.diagram_generator import generate_diagram

text = """
1. Начало
2. Проверить условие
3. Если да — действие A
4. Если нет — действие B
5. Конец
"""
plantuml_source, png_path = generate_diagram(text, output_format="png", use_gpu=False)
print(plantuml_source)
# png_path — путь к временному PNG или None, если рендер не удался
```

Для рендера PNG используется пакет **plantuml** (удалённый сервер PlantUML).

### Проверка бэкенда (Transformers / llama.cpp)

```powershell
.venv\Scripts\python.exe -c "from src.diagram_extractor import _detect_backend; print('Backend:', _detect_backend())"
```

### Запуск тестов

```powershell
.venv\Scripts\python.exe tests\test_qwen_integration.py
```

---

## Поддерживаемые форматы

| Расширение | Обработка |
|------------|-----------|
| `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` | Прямая отправка в VLM (Qwen2.5-VL) |
| `.bpmn`, `.drawio` | Парсинг XML → текст «Шаг \| Роль» без VLM |
| `.xml` | Если корень тега BPMN — как `.bpmn` |
| `.svg` | Конвертация в PNG (cairosvg) → VLM |
| `.uml`, `.puml` | Рендер PlantUML в PNG (опционально `plantuml`) → VLM |

Точный список расширений возвращает **GET /formats**.

Для SVG нужен **cairosvg** (уже в `requirements.txt`). Для `.uml`/`.puml` опционально: `pip install plantuml` (используется удалённый сервер PlantUML).

---

## Бэкенды (Transformers / llama.cpp)

- **Transformers** (по умолчанию): Hugging Face `Qwen2.5-VL-3B-Instruct`. Работает на Windows/Linux, CPU/GPU. Модель скачивается при первом использовании.
- **llama-cpp-python**: GGUF-модели (например, Mungert/Qwen2.5-VL-3B-Instruct-GGUF). Меньше потребление RAM при квантизации, удобно для Docker/Linux. Если пакет установлен, он выбирается первым (приоритет над Transformers).

### Настройка llama.cpp с Qwen2.5-VL

Для работы с изображениями нужны два файла: основная GGUF-модель и mmproj (визуальный энкодер).

**Ошибка «ninja: build stopped» / «Failed building wheel»:** не собирайте из исходников — используйте предсобранный wheel (см. ниже).

#### 1. Установка llama-cpp-python

**Только CPU (рекомендуется для Windows без сборки):**

```powershell
.venv\Scripts\pip uninstall llama-cpp-python -y
.venv\Scripts\pip install llama-cpp-python --only-binary=llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

**С поддержкой CUDA** (Python 3.10–3.12, CUDA 12.x; подставьте свою версию: `cu121`, `cu122`, `cu123`, `cu124`, `cu125`):

```powershell
.venv\Scripts\pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

На Linux/macOS обычно `pip install llama-cpp-python` собирается из исходников. Для GPU:  
`CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python`.

#### 2. Скачивание моделей

Из корня проекта:

```powershell
.venv\Scripts\python.exe scripts\download_models.py
```

Скрипт создаёт каталог `models/` и загружает:

- `Qwen2.5-VL-3B-Instruct-q4_k_m.gguf` (~2 GB)
- `Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf` (~1.3 GB)

Другая квантизация (больше точность, больше RAM):

```powershell
.venv\Scripts\python.exe scripts\download_models.py --quant q8_0
```

Для нестандартных путей задайте переменные окружения перед запуском:

- `LLAMA_MODEL_PATH` — путь к файлу GGUF модели.
- `LLAMA_MMPROJ_PATH` — путь к файлу mmproj.

Если в `models/` лежат файлы с именами по умолчанию (q4_k_m и mmproj-f16), они подхватываются автоматически.

#### 3. Проверка бэкенда

```powershell
.venv\Scripts\python.exe -c "from src.diagram_extractor import _detect_backend; print('Backend:', _detect_backend())"
```

Должно вывести `Backend: llama_cpp`, если llama-cpp-python установлен и модели на месте.

---

## Структура проекта

```
diagrams/
├── models/                    # GGUF + mmproj для llama.cpp (создаётся download_models.py)
├── src/
│   ├── api.py                 # FastAPI: /extract, /generate-diagram, /docs, /health
│   ├── diagram_extractor.py   # Извлечение: диаграмма → текст (VLM, BPMN, drawio)
│   ├── diagram_formats.py     # Парсинг .bpmn/.drawio, конвертация .svg/.uml, рендер PlantUML
│   ├── image_preprocessing.py # Препроцессинг изображений перед VLM (апскейл, контраст)
│   ├── diagram_generator.py   # Генерация: текст алгоритма → PlantUML → PNG
│   └── __init__.py
├── scripts/
│   └── download_models.py     # Скачивание GGUF + mmproj (llama.cpp)
├── tests/
│   ├── test_qwen_integration.py
│   └── test_formats.py
├── requirements.txt
└── README.md
```

---

## Устранение неполадок

- **Сервер запустился, но первый запрос к /extract или /generate-diagram очень долгий** — идёт загрузка модели с Hugging Face. Дождитесь завершения или включите предзагрузку (она уже в lifespan при старте uvicorn).
- **Out of memory (OOM)** — используйте CPU (`USE_GPU` не задавать или `false`), либо llama.cpp с квантизацией q4_k_m и предсобранным CPU-wheel.
- **PNG не возвращается в /generate-diagram** — проверьте доступ в интернет к `https://www.plantuml.com/plantuml`. В ответе смотрите `render_note`.
- **«Неподдерживаемый формат» в /extract** — проверьте список в **GET /formats** и расширение загружаемого файла.
- **Windows: python не найден** — используйте `py` или полный путь `.venv\Scripts\python.exe` / `.venv\Scripts\uvicorn.exe`.