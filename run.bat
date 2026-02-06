@echo off
REM Diagram Converter - Run web app and API
REM Double-click or: run.bat
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 py -m venv .venv
)

call .venv\Scripts\activate.bat

echo.
echo Diagram Converter - Web UI: http://127.0.0.1:8000/
echo API docs: http://127.0.0.1:8000/docs
echo.

.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000
pause
