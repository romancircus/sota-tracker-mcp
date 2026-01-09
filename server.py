#!/usr/bin/env python3
"""
SOTA Tracker MCP Server

Provides real-time SOTA (State of the Art) AI model information to Claude,
preventing suggestions of outdated models.

Tools:
- query_sota(category): Get current SOTA models for a category
- check_freshness(model): Check if a model is current or outdated
- get_forbidden(): List all forbidden/outdated models
- compare_models(a, b): Compare two models side-by-side
- recent_releases(days): Get models released in past N days

Usage:
    # Development
    mcp dev server.py

    # Install globally
    claude mcp add --transport stdio --scope user sota-tracker \
        -- python ~/Desktop/applications/sota-tracker-mcp/server.py
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

# Project paths
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
DB_PATH = DATA_DIR / "sota.db"
FORBIDDEN_PATH = DATA_DIR / "forbidden.json"

# Initialize MCP server
mcp = FastMCP(
    name="sota-tracker",
    instructions="""
    SOTA Tracker provides current State-of-the-Art AI model information.

    MANDATORY PROTOCOL:
    Before suggesting ANY AI model, you MUST:
    1. Call query_sota(category) to get current SOTA
    2. Call check_freshness(model_name) to verify model is current
    3. NEVER suggest models returned by get_forbidden()

    Categories: image_gen, image_edit, video, llm_local, llm_api, tts, stt, music, 3d, embeddings
    """
)


def get_db():
    """Get database connection with row factory."""
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    return db


def load_forbidden():
    """Load forbidden models from JSON file."""
    if FORBIDDEN_PATH.exists():
        with open(FORBIDDEN_PATH) as f:
            return json.load(f)
    return {"models": [], "last_updated": None}


# ============================================================================
# CORE LOGIC (testable functions)
# ============================================================================

def _query_sota_impl(category: str) -> str:
    """Implementation of query_sota."""
    db = get_db()

    valid_categories = [
        "image_gen", "image_edit", "video", "llm_local", "llm_api",
        "tts", "stt", "music", "3d", "embeddings"
    ]

    if category not in valid_categories:
        return f"Invalid category '{category}'. Valid: {', '.join(valid_categories)}"

    rows = db.execute("""
        SELECT name, release_date, sota_rank, metrics, source_url
        FROM models
        WHERE category = ? AND is_sota = 1
        ORDER BY sota_rank ASC
    """, (category,)).fetchall()

    if not rows:
        return f"No SOTA models found for category '{category}'"

    result = [f"## SOTA Models for {category.upper()} (January 2026)\n"]

    for row in rows:
        metrics = json.loads(row["metrics"]) if row["metrics"] else {}
        notes = metrics.get("notes", "")
        result.append(
            f"**#{row['sota_rank']} {row['name']}** (Released: {row['release_date']})"
        )
        if notes:
            result.append(f"   - {notes}")

    return "\n".join(result)


def _check_freshness_impl(model_name: str) -> str:
    """Implementation of check_freshness."""
    forbidden = load_forbidden()
    for model in forbidden.get("models", []):
        if model["name"].lower() == model_name.lower():
            return (
                f"OUTDATED: {model['name']} - {model['reason']}\n"
                f"USE INSTEAD: {model['replacement']}"
            )

    db = get_db()
    row = db.execute("""
        SELECT name, category, is_sota, sota_rank, release_date
        FROM models
        WHERE LOWER(name) = LOWER(?)
    """, (model_name,)).fetchone()

    if row:
        if row["is_sota"]:
            return f"CURRENT: {row['name']} is #{row['sota_rank']} SOTA for {row['category']} (Released: {row['release_date']})"
        else:
            sota = db.execute("""
                SELECT name FROM models
                WHERE category = ? AND is_sota = 1 AND sota_rank = 1
            """, (row["category"],)).fetchone()

            replacement = sota["name"] if sota else "Check query_sota() for current options"
            return f"OUTDATED: {row['name']} is no longer SOTA.\nUSE INSTEAD: {replacement}"

    return f"UNKNOWN: Model '{model_name}' not in database. Use query_sota() to find current options."


def _get_forbidden_impl() -> str:
    """Implementation of get_forbidden."""
    forbidden = load_forbidden()
    models = forbidden.get("models", [])

    if not models:
        return "No forbidden models defined."

    result = [f"## Forbidden Models (Do Not Suggest)\n"]
    result.append(f"Last updated: {forbidden.get('last_updated', 'Unknown')}\n")

    by_category = {}
    for m in models:
        cat = m.get("category", "other")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(m)

    for category, items in sorted(by_category.items()):
        result.append(f"\n### {category.upper()}")
        for m in items:
            result.append(f"- **{m['name']}**: {m['reason']}")
            result.append(f"  → Use: {m['replacement']}")

    return "\n".join(result)


def _compare_models_impl(model_a: str, model_b: str) -> str:
    """Implementation of compare_models."""
    db = get_db()

    def get_model_info(name: str) -> Optional[dict]:
        row = db.execute("""
            SELECT name, category, release_date, is_sota, sota_rank, metrics
            FROM models WHERE LOWER(name) = LOWER(?)
        """, (name,)).fetchone()
        if row:
            return dict(row)

        forbidden = load_forbidden()
        for m in forbidden.get("models", []):
            if m["name"].lower() == name.lower():
                return {
                    "name": m["name"],
                    "category": m.get("category", "unknown"),
                    "is_sota": False,
                    "sota_rank": None,
                    "status": "FORBIDDEN",
                    "reason": m["reason"],
                    "replacement": m["replacement"]
                }
        return None

    info_a = get_model_info(model_a)
    info_b = get_model_info(model_b)

    if not info_a and not info_b:
        return f"Neither '{model_a}' nor '{model_b}' found in database."

    result = ["## Model Comparison\n"]

    def format_model(info: Optional[dict], name: str) -> str:
        if not info:
            return f"**{name}**: Not in database"

        lines = [f"**{info['name']}**"]
        lines.append(f"- Category: {info.get('category', 'Unknown')}")

        if info.get("status") == "FORBIDDEN":
            lines.append(f"- Status: FORBIDDEN")
            lines.append(f"- Reason: {info.get('reason', 'Unknown')}")
            lines.append(f"- Use instead: {info.get('replacement', 'Unknown')}")
        elif info.get("is_sota"):
            lines.append(f"- Status: SOTA #{info.get('sota_rank', '?')}")
            lines.append(f"- Released: {info.get('release_date', 'Unknown')}")
        else:
            lines.append(f"- Status: Not SOTA")
            lines.append(f"- Released: {info.get('release_date', 'Unknown')}")

        return "\n".join(lines)

    result.append(format_model(info_a, model_a))
    result.append("\n---\n")
    result.append(format_model(info_b, model_b))

    result.append("\n### Recommendation")
    if info_a and info_a.get("status") == "FORBIDDEN":
        result.append(f"Use {info_b['name'] if info_b else model_b} (or its SOTA replacement)")
    elif info_b and info_b.get("status") == "FORBIDDEN":
        result.append(f"Use {info_a['name'] if info_a else model_a} (or its SOTA replacement)")
    elif info_a and info_a.get("is_sota") and (not info_b or not info_b.get("is_sota")):
        result.append(f"Use {info_a['name']} (currently SOTA)")
    elif info_b and info_b.get("is_sota") and (not info_a or not info_a.get("is_sota")):
        result.append(f"Use {info_b['name']} (currently SOTA)")
    else:
        result.append("Both models are comparable. Check query_sota() for latest rankings.")

    return "\n".join(result)


def _recent_releases_impl(days: int = 30) -> str:
    """Implementation of recent_releases."""
    db = get_db()

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    rows = db.execute("""
        SELECT name, category, release_date, is_sota, sota_rank
        FROM models
        WHERE release_date >= ?
        ORDER BY release_date DESC
    """, (cutoff,)).fetchall()

    if not rows:
        return f"No models released in the past {days} days."

    result = [f"## Models Released in Past {days} Days\n"]

    for row in rows:
        sota_badge = f" [SOTA #{row['sota_rank']}]" if row["is_sota"] else ""
        result.append(f"- **{row['name']}** ({row['category']}) - {row['release_date']}{sota_badge}")

    return "\n".join(result)


# ============================================================================
# MCP TOOLS (wrappers for MCP decorator)
# ============================================================================

@mcp.tool()
def query_sota(category: str) -> str:
    """
    Get current SOTA (State of the Art) models for a category.

    Args:
        category: One of: image_gen, image_edit, video, llm_local, llm_api,
                  tts, stt, music, 3d, embeddings

    Returns:
        Ranked list of current SOTA models with release dates and notes.
    """
    return _query_sota_impl(category)


@mcp.tool()
def check_freshness(model_name: str) -> str:
    """
    Check if a model is current SOTA or outdated.

    Args:
        model_name: Name of the model to check (e.g., "FLUX.1-dev", "Llama 2")

    Returns:
        Status: "CURRENT" with rank, or "OUTDATED" with replacement suggestion.
    """
    return _check_freshness_impl(model_name)


@mcp.tool()
def get_forbidden() -> str:
    """
    Get list of all forbidden/outdated models that should never be suggested.

    Returns:
        List of outdated models with reasons and replacements.
    """
    return _get_forbidden_impl()


@mcp.tool()
def compare_models(model_a: str, model_b: str) -> str:
    """
    Compare two models side-by-side.

    Args:
        model_a: First model name
        model_b: Second model name

    Returns:
        Side-by-side comparison of both models.
    """
    return _compare_models_impl(model_a, model_b)


@mcp.tool()
def recent_releases(days: int = 30) -> str:
    """
    Get models released in the past N days.

    Args:
        days: Number of days to look back (default: 30)

    Returns:
        List of recently released models.
    """
    return _recent_releases_impl(days)


# ============================================================================
# MCP RESOURCES
# ============================================================================

@mcp.resource("sota://categories")
def get_categories() -> str:
    """List all available model categories."""
    categories = {
        "image_gen": "Text/image to image generation (FLUX, Qwen-Image, Z-Image)",
        "image_edit": "Image editing and inpainting (Qwen-Image-Edit, Kontext)",
        "video": "Video generation (LTX-2, Wan2.1, HunyuanVideo)",
        "llm_local": "Local LLMs for llama.cpp (Qwen2.5, Llama3.3, DeepSeek)",
        "llm_api": "API-based LLMs (Claude, GPT, Gemini)",
        "tts": "Text-to-speech (ChatterboxTTS, F5-TTS)",
        "stt": "Speech-to-text (Whisper)",
        "music": "Music generation (Suno, Udio)",
        "3d": "3D model generation (Meshy, Tripo)",
        "embeddings": "Vector embeddings (BGE, E5, Cohere)"
    }

    result = ["# Model Categories\n"]
    for cat_id, desc in categories.items():
        result.append(f"- **{cat_id}**: {desc}")

    return "\n".join(result)


@mcp.resource("sota://forbidden")
def get_forbidden_resource() -> str:
    """Get the complete forbidden models list."""
    return _get_forbidden_impl()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
