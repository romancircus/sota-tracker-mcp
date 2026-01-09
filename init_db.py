#!/usr/bin/env python3
"""
Initialize the SOTA Tracker database with schema and seed data.

Run this once to set up the database:
    python init_db.py
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
DB_PATH = DATA_DIR / "sota.db"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)


def create_schema(db: sqlite3.Connection):
    """Create database schema."""
    db.executescript("""
        -- Models table
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            release_date DATE,
            source TEXT DEFAULT 'manual',
            source_url TEXT,
            is_sota BOOLEAN DEFAULT FALSE,
            sota_rank INTEGER,
            metrics JSON,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Forbidden models (also in JSON for easy editing)
        CREATE TABLE IF NOT EXISTS forbidden (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            reason TEXT NOT NULL,
            replacement TEXT,
            category TEXT,
            deprecated_date DATE,
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Categories
        CREATE TABLE IF NOT EXISTS categories (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT
        );

        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_models_category ON models(category);
        CREATE INDEX IF NOT EXISTS idx_models_sota ON models(is_sota);
        CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
    """)


def seed_categories(db: sqlite3.Connection):
    """Seed model categories."""
    categories = [
        ("image_gen", "Image Generation", "Text/image to image generation"),
        ("image_edit", "Image Editing", "Image editing, inpainting, style transfer"),
        ("video", "Video Generation", "Text/image to video"),
        ("llm_local", "Local LLMs", "LLMs for local inference (GGUF, llama.cpp)"),
        ("llm_api", "API LLMs", "Cloud-based LLM APIs"),
        ("tts", "Text-to-Speech", "Voice synthesis and cloning"),
        ("stt", "Speech-to-Text", "Audio transcription"),
        ("music", "Music Generation", "AI music creation"),
        ("3d", "3D Generation", "3D model and scene generation"),
        ("embeddings", "Embeddings", "Vector embedding models"),
    ]

    db.executemany(
        "INSERT OR REPLACE INTO categories (id, name, description) VALUES (?, ?, ?)",
        categories
    )


def seed_sota_models(db: sqlite3.Connection):
    """Seed current SOTA models (January 2026)."""

    models = [
        # Image Generation
        {
            "id": "qwen-image-2512",
            "name": "Qwen-Image-2512",
            "category": "image_gen",
            "release_date": "2025-12-31",
            "is_sota": True,
            "sota_rank": 1,
            "metrics": {"notes": "Best human realism, top tier quality"}
        },
        {
            "id": "z-image-turbo",
            "name": "Z-Image-Turbo",
            "category": "image_gen",
            "release_date": "2025-11-26",
            "is_sota": True,
            "sota_rank": 2,
            "metrics": {"notes": "Fastest, #1 open source, Apache 2.0"}
        },
        {
            "id": "flux2-dev",
            "name": "FLUX.2-dev",
            "category": "image_gen",
            "release_date": "2025-11-25",
            "is_sota": True,
            "sota_rank": 3,
            "metrics": {"notes": "Multi-reference support (up to 10 images)"}
        },

        # Image Editing
        {
            "id": "qwen-image-edit-2511",
            "name": "Qwen-Image-Edit-2511",
            "category": "image_edit",
            "release_date": "2025-12-01",
            "is_sota": True,
            "sota_rank": 1,
            "metrics": {"notes": "Instruction-based editing, best quality"}
        },
        {
            "id": "flux1-kontext",
            "name": "FLUX.1-Kontext",
            "category": "image_edit",
            "release_date": "2025-06-01",
            "is_sota": True,
            "sota_rank": 2,
            "metrics": {"notes": "Color/style editing, good for recoloring"}
        },

        # Video Generation
        {
            "id": "ltx-2",
            "name": "LTX-2",
            "category": "video",
            "release_date": "2026-01-06",
            "is_sota": True,
            "sota_rank": 1,
            "metrics": {"notes": "Audio support, 18x faster, 20s duration, 4K"}
        },
        {
            "id": "wan-2.6",
            "name": "Wan 2.6",
            "category": "video",
            "release_date": "2025-12-16",
            "is_sota": True,
            "sota_rank": 2,
            "metrics": {"notes": "Audio + reference-to-video, multi-shot"}
        },
        {
            "id": "hunyuan-video-1.5",
            "name": "HunyuanVideo 1.5",
            "category": "video",
            "release_date": "2025-11-21",
            "is_sota": True,
            "sota_rank": 3,
            "metrics": {"notes": "Best for multi-character scenes, 16GB VRAM"}
        },
        {
            "id": "wan2.1-i2v",
            "name": "Wan2.1-I2V",
            "category": "video",
            "release_date": "2025-06-01",
            "is_sota": True,
            "sota_rank": 4,
            "metrics": {"notes": "Best creature realism, silky motion, detailed textures"}
        },

        # Local LLMs
        {
            "id": "qwen2.5-72b",
            "name": "Qwen2.5-72B",
            "category": "llm_local",
            "release_date": "2025-09-01",
            "is_sota": True,
            "sota_rank": 1,
            "metrics": {"notes": "Best quality, 41GB GGUF"}
        },
        {
            "id": "llama3.3-70b",
            "name": "Llama3.3-70B",
            "category": "llm_local",
            "release_date": "2025-12-01",
            "is_sota": True,
            "sota_rank": 2,
            "metrics": {"notes": "General purpose, 40GB GGUF"}
        },
        {
            "id": "deepseek-v3",
            "name": "DeepSeek-V3",
            "category": "llm_local",
            "release_date": "2025-12-25",
            "is_sota": True,
            "sota_rank": 3,
            "metrics": {"notes": "Best for coding, 37GB GGUF"}
        },
        {
            "id": "qwen2.5-32b",
            "name": "Qwen2.5-32B",
            "category": "llm_local",
            "release_date": "2025-09-01",
            "is_sota": True,
            "sota_rank": 4,
            "metrics": {"notes": "Balanced quality/speed, 19GB GGUF"}
        },
        {
            "id": "qwen2.5-7b",
            "name": "Qwen2.5-7B",
            "category": "llm_local",
            "release_date": "2025-09-01",
            "is_sota": True,
            "sota_rank": 5,
            "metrics": {"notes": "Fast default, 4.4GB GGUF"}
        },

        # API LLMs
        {
            "id": "claude-opus-4.5",
            "name": "Claude Opus 4.5",
            "category": "llm_api",
            "release_date": "2025-11-01",
            "is_sota": True,
            "sota_rank": 1,
            "metrics": {"notes": "Best reasoning, Anthropic"}
        },
        {
            "id": "gpt-4.5",
            "name": "GPT-4.5",
            "category": "llm_api",
            "release_date": "2025-10-01",
            "is_sota": True,
            "sota_rank": 2,
            "metrics": {"notes": "Strong multimodal, OpenAI"}
        },
        {
            "id": "gemini-2.0-pro",
            "name": "Gemini 2.0 Pro",
            "category": "llm_api",
            "release_date": "2025-12-01",
            "is_sota": True,
            "sota_rank": 3,
            "metrics": {"notes": "Long context, Google"}
        },

        # TTS
        {
            "id": "chatterbox-tts",
            "name": "ChatterboxTTS",
            "category": "tts",
            "release_date": "2025-06-01",
            "is_sota": True,
            "sota_rank": 1,
            "metrics": {"notes": "Best open-source TTS"}
        },
        {
            "id": "f5-tts",
            "name": "F5-TTS",
            "category": "tts",
            "release_date": "2025-08-01",
            "is_sota": True,
            "sota_rank": 2,
            "metrics": {"notes": "Voice cloning specialist"}
        },

        # STT
        {
            "id": "whisper-large-v3",
            "name": "Whisper Large v3",
            "category": "stt",
            "release_date": "2024-11-01",
            "is_sota": True,
            "sota_rank": 1,
            "metrics": {"notes": "Best accuracy, OpenAI"}
        },

        # Music
        {
            "id": "suno-v4",
            "name": "Suno v4",
            "category": "music",
            "release_date": "2025-09-01",
            "is_sota": True,
            "sota_rank": 1,
            "metrics": {"notes": "Best quality (API)"}
        },

        # Embeddings
        {
            "id": "bge-m3",
            "name": "BGE-M3",
            "category": "embeddings",
            "release_date": "2024-06-01",
            "is_sota": True,
            "sota_rank": 1,
            "metrics": {"notes": "Multilingual, dense+sparse+colbert"}
        },
    ]

    for model in models:
        db.execute("""
            INSERT OR REPLACE INTO models
            (id, name, category, release_date, is_sota, sota_rank, metrics, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'manual')
        """, (
            model["id"],
            model["name"],
            model["category"],
            model["release_date"],
            model["is_sota"],
            model["sota_rank"],
            json.dumps(model.get("metrics", {}))
        ))


def main():
    """Initialize the database."""
    print(f"Initializing database at {DB_PATH}")

    # Remove existing database for fresh start
    if DB_PATH.exists():
        DB_PATH.unlink()
        print("Removed existing database")

    db = sqlite3.connect(str(DB_PATH))

    print("Creating schema...")
    create_schema(db)

    print("Seeding categories...")
    seed_categories(db)

    print("Seeding SOTA models...")
    seed_sota_models(db)

    db.commit()
    db.close()

    print(f"Database initialized successfully at {DB_PATH}")
    print(f"Size: {DB_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
