#!/usr/bin/env python3
"""
Скачивание GGUF-моделей Qwen2.5-VL-3B для бэкенда llama-cpp-python.
После загрузки сервис автоматически использует файлы из каталога models/
если установлен llama-cpp-python.

Использование:
  python scripts/download_models.py
  python scripts/download_models.py --quant q8_0   # другая квантизация (больше RAM)
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Mungert/Qwen2.5-VL-3B-Instruct-GGUF"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MMPROJ_FILENAME = "Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf"

# Доступные квантизации в репозитории (файл ~2 GB для q4_k_m)
QUANT_FILENAMES = {
    "q4_k_m": "Qwen2.5-VL-3B-Instruct-q4_k_m.gguf",
    "q4_k_s": "Qwen2.5-VL-3B-Instruct-q4_k_s.gguf",
    "q5_k_m": "Qwen2.5-VL-3B-Instruct-q5_k_m.gguf",
    "q8_0": "Qwen2.5-VL-3B-Instruct-q8_0.gguf",
    "q4_0": "Qwen2.5-VL-3B-Instruct-q4_0.gguf",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen2.5-VL-3B GGUF for llama.cpp")
    parser.add_argument(
        "--quant",
        default="q4_k_m",
        choices=list(QUANT_FILENAMES),
        help="Квантизация модели (q4_k_m — по умолчанию, меньше RAM)",
    )
    args = parser.parse_args()
    llm_filename = QUANT_FILENAMES[args.quant]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {MODELS_DIR}")
    print(f"LLM: {llm_filename}")
    print(f"mmproj: {MMPROJ_FILENAME}")
    print("This may take several minutes...")

    llm_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=llm_filename,
        local_dir=MODELS_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded LLM: {llm_path}")

    mmproj_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MMPROJ_FILENAME,
        local_dir=MODELS_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded mmproj: {mmproj_path}")
    print("Done. Restart the app to use llama.cpp with local models.")
    if args.quant != "q4_k_m":
        print(f"Set env LLAMA_MODEL_PATH={llm_path} (or use default q4_k_m in models/)")
