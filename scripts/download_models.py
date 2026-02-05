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

# Доступные квантизации (размер ~2 GB q4_k_m … ~4 GB q8_0 / f16-q8_0; в 7GB VRAM укладывается f16-q8_0)
QUANT_FILENAMES = {
    "q4_0": "Qwen2.5-VL-3B-Instruct-q4_0.gguf",
    "q4_k_s": "Qwen2.5-VL-3B-Instruct-q4_k_s.gguf",
    "q4_k_m": "Qwen2.5-VL-3B-Instruct-q4_k_m.gguf",
    "q5_k_m": "Qwen2.5-VL-3B-Instruct-q5_k_m.gguf",
    "q8_0": "Qwen2.5-VL-3B-Instruct-q8_0.gguf",
    "f16-q8_0": "Qwen2.5-VL-3B-Instruct-f16-q8_0.gguf",
    "bf16-q8_0": "Qwen2.5-VL-3B-Instruct-bf16-q8_0.gguf",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen2.5-VL-3B GGUF for llama.cpp")
    parser.add_argument(
        "--quant",
        default="q8_0",
        choices=list(QUANT_FILENAMES),
        help="Квантизация (q8_0 по умолчанию; q4_k_m — экономия RAM)",
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
    if args.quant != "q8_0":
        print(f"Set env LLAMA_QUANT={args.quant} (or use default q8_0 in app)")
