#!/usr/bin/env python3
"""
SOTA Tracker MCP Server

Provides real-time SOTA (State of the Art) AI model information to Claude,
preventing suggestions of outdated models.

Features:
- Defaults to open-source/local models (but aware of closed-source)
- Filter by open-source only
- 11 categories including coding LLMs

Tools:
- query_sota(category, open_source_only): Get current SOTA models
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

from utils.db import get_db as _get_db

# Smart caching - refresh data if stale (not fetched today)
try:
    from fetchers.cache_manager import get_cache_manager
    CACHE_ENABLED = True
except ImportError:
    CACHE_ENABLED = False

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

    DEFAULT BEHAVIOR: Shows open-source/local models first (but is aware of closed-source).

    MANDATORY PROTOCOL:
    Before suggesting ANY AI model, you MUST:
    1. Call query_sota(category) to get current SOTA (defaults to open-source)
    2. Call check_freshness(model_name) to verify model is current
    3. NEVER suggest models returned by get_forbidden()

    Categories:
    - image_gen, image_edit, video
    - llm_local (general local LLMs), llm_api (cloud APIs), llm_coding (code-focused)
    - tts, stt, music, 3d, embeddings

    Use open_source_only=False to see all models including closed-source.
    """
)


def get_db():
    """Get database connection with row factory."""
    return _get_db(DB_PATH)


def load_forbidden():
    """Load forbidden models from JSON file."""
    if FORBIDDEN_PATH.exists():
        with open(FORBIDDEN_PATH) as f:
            return json.load(f)
    return {"models": [], "last_updated": None}


# ============================================================================
# CORE LOGIC (testable functions)
# ============================================================================

def _query_sota_impl(category: str, open_source_only: bool = True) -> str:
    """Implementation of query_sota."""

    valid_categories = [
        "image_gen", "image_edit", "video", "llm_local", "llm_api", "llm_coding",
        "tts", "stt", "music", "3d", "embeddings"
    ]

    if category not in valid_categories:
        return f"Invalid category '{category}'. Valid: {', '.join(valid_categories)}"

    # Smart cache: refresh if data is stale (not fetched today)
    refresh_note = ""
    if CACHE_ENABLED:
        try:
            cache_mgr = get_cache_manager(DB_PATH)
            result = cache_mgr.refresh_if_stale(category)
            if result["refreshed"]:
                refresh_note = f"\n[Data refreshed from {result['source']}, {result['models_updated']} models updated]\n"
            elif result["error"]:
                refresh_note = f"\n[Cache refresh failed: {result['error']}. Using cached data.]\n"
        except Exception as e:
            refresh_note = f"\n[Cache error: {e}. Using cached data.]\n"

    db = get_db()

    # Build query based on open_source_only flag
    # Get current month/year for header
    current_date = datetime.now().strftime("%B %Y")

    if open_source_only:
        rows = db.execute("""
            SELECT name, release_date, sota_rank_open as rank, metrics, source_url, is_open_source
            FROM models
            WHERE category = ? AND is_sota = 1 AND is_open_source = 1
            ORDER BY sota_rank_open ASC
        """, (category,)).fetchall()
        header = f"## SOTA Open-Source Models for {category.upper()} ({current_date})\n"
    else:
        rows = db.execute("""
            SELECT name, release_date, sota_rank as rank, metrics, source_url, is_open_source
            FROM models
            WHERE category = ? AND is_sota = 1
            ORDER BY sota_rank ASC
        """, (category,)).fetchall()
        header = f"## ALL SOTA Models for {category.upper()} ({current_date})\n"

    if not rows:
        if open_source_only:
            return f"No open-source SOTA models found for '{category}'. Try open_source_only=False to see closed-source options."
        return f"No SOTA models found for category '{category}'"

    result = [header]

    if refresh_note:
        result.append(refresh_note)

    if open_source_only:
        result.append("(Showing open-source only. Use open_source_only=False for all models.)\n")

    for row in rows:
        try:
            metrics = json.loads(row["metrics"]) if row["metrics"] else {}
        except json.JSONDecodeError:
            metrics = {}
        notes = metrics.get("notes", "")
        why_sota = metrics.get("why_sota", "")
        strengths = metrics.get("strengths", [])
        use_cases = metrics.get("use_cases", [])

        # Add badge for open/closed
        badge = "" if row["is_open_source"] else " [CLOSED]"

        result.append(
            f"**#{row['rank']} {row['name']}**{badge} (Released: {row['release_date']})"
        )
        if notes:
            result.append(f"   {notes}")
        if why_sota:
            result.append(f"   Why SOTA: {why_sota}")
        if strengths:
            result.append(f"   Strengths: {', '.join(strengths)}")
        if use_cases:
            result.append(f"   Best for: {', '.join(use_cases)}")

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
        SELECT name, category, is_sota, is_open_source, sota_rank, sota_rank_open, release_date
        FROM models
        WHERE LOWER(name) = LOWER(?)
    """, (model_name,)).fetchone()

    if row:
        open_badge = "open-source" if row["is_open_source"] else "closed-source"
        if row["is_sota"]:
            rank = row["sota_rank_open"] if row["is_open_source"] else row["sota_rank"]
            rank_type = "open-source" if row["is_open_source"] else "overall"
            return f"CURRENT: {row['name']} is #{rank} {rank_type} SOTA for {row['category']} ({open_badge}, Released: {row['release_date']})"
        else:
            # Find current SOTA for same category
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
            SELECT name, category, release_date, is_sota, is_open_source, sota_rank, sota_rank_open, metrics
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

        # Open source status
        if 'is_open_source' in info:
            lines.append(f"- License: {'Open-source' if info['is_open_source'] else 'Closed-source'}")

        if info.get("status") == "FORBIDDEN":
            lines.append(f"- Status: FORBIDDEN")
            lines.append(f"- Reason: {info.get('reason', 'Unknown')}")
            lines.append(f"- Use instead: {info.get('replacement', 'Unknown')}")
        elif info.get("is_sota"):
            rank = info.get('sota_rank_open') or info.get('sota_rank', '?')
            lines.append(f"- Status: SOTA #{rank}")
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
    elif info_a and info_b:
        # Both are SOTA, recommend based on open-source preference
        if info_a.get("is_open_source") and not info_b.get("is_open_source"):
            result.append(f"Use {info_a['name']} (open-source, preferred for local deployment)")
        elif info_b.get("is_open_source") and not info_a.get("is_open_source"):
            result.append(f"Use {info_b['name']} (open-source, preferred for local deployment)")
        else:
            result.append("Both models are comparable. Check query_sota() for latest rankings.")
    else:
        result.append("Both models are comparable. Check query_sota() for latest rankings.")

    return "\n".join(result)


def _recent_releases_impl(days: int = 30, open_source_only: bool = True) -> str:
    """Implementation of recent_releases."""
    # Input validation
    if not isinstance(days, int) or days < 1 or days > 365:
        return "Error: days must be an integer between 1 and 365"

    db = get_db()

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    if open_source_only:
        rows = db.execute("""
            SELECT name, category, release_date, is_sota, sota_rank_open as rank, is_open_source
            FROM models
            WHERE release_date >= ? AND is_open_source = 1
            ORDER BY release_date DESC
        """, (cutoff,)).fetchall()
    else:
        rows = db.execute("""
            SELECT name, category, release_date, is_sota, sota_rank as rank, is_open_source
            FROM models
            WHERE release_date >= ?
            ORDER BY release_date DESC
        """, (cutoff,)).fetchall()

    if not rows:
        return f"No models released in the past {days} days."

    header = f"## Models Released in Past {days} Days"
    if open_source_only:
        header += " (Open-Source Only)"
    result = [header + "\n"]

    for row in rows:
        sota_badge = f" [SOTA #{row['rank']}]" if row["is_sota"] and row["rank"] else ""
        open_badge = "" if row["is_open_source"] else " [CLOSED]"
        result.append(f"- **{row['name']}** ({row['category']}) - {row['release_date']}{sota_badge}{open_badge}")

    return "\n".join(result)


# ============================================================================
# MCP TOOLS (wrappers for MCP decorator)
# ============================================================================

@mcp.tool()
def query_sota(category: str, open_source_only: bool = True) -> str:
    """
    Get current SOTA (State of the Art) models for a category.

    DEFAULTS TO OPEN-SOURCE ONLY. Set open_source_only=False to see closed-source models too.

    Args:
        category: One of: image_gen, image_edit, video, llm_local, llm_api, llm_coding,
                  tts, stt, music, 3d, embeddings
        open_source_only: If True (default), only show open-source models.
                          If False, show all models including closed-source.

    Returns:
        Ranked list of current SOTA models with release dates and notes.
    """
    return _query_sota_impl(category, open_source_only)


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
        Side-by-side comparison of both models including open-source status.
    """
    return _compare_models_impl(model_a, model_b)


@mcp.tool()
def recent_releases(days: int = 30, open_source_only: bool = True) -> str:
    """
    Get models released in the past N days.

    Args:
        days: Number of days to look back (default: 30)
        open_source_only: If True (default), only show open-source models.

    Returns:
        List of recently released models.
    """
    return _recent_releases_impl(days, open_source_only)


@mcp.tool()
def refresh_data(category: str = None) -> str:
    """
    Force refresh SOTA data from external sources.

    By default, data auto-refreshes once per day on first query.
    Use this to force an immediate refresh.

    Args:
        category: Category to refresh, or None for all categories.

    Returns:
        Refresh status for each category.
    """
    if not CACHE_ENABLED:
        return "Cache system not available. Using static data."

    try:
        cache_mgr = get_cache_manager(DB_PATH)
        results = []

        if category:
            # Refresh single category
            result = cache_mgr.force_refresh(category)
            status = "OK" if result["refreshed"] else f"FAILED: {result['error']}"
            return f"Refresh {category}: {status} (source: {result['source']}, updated: {result['models_updated']} models)"
        else:
            # Refresh all categories
            categories = [
                "llm_local", "llm_api", "llm_coding",
                "image_gen", "video", "tts"
            ]
            for cat in categories:
                result = cache_mgr.force_refresh(cat)
                status = "OK" if result["refreshed"] else f"FAILED"
                results.append(f"  {cat}: {status} ({result['models_updated']} models)")

            return "## Refresh Results\n" + "\n".join(results)
    except Exception as e:
        return f"Refresh error: {e}"


@mcp.tool()
def cache_status() -> str:
    """
    Check the cache status for all categories.

    Shows when each category was last refreshed and from what source.

    Returns:
        Cache freshness status for all categories.
    """
    if not CACHE_ENABLED:
        return "Cache system not available. Using static data."

    try:
        cache_mgr = get_cache_manager(DB_PATH)
        statuses = cache_mgr.get_cache_status()

        if not statuses:
            return "No cache data yet. Categories will be refreshed on first query."

        result = ["## Cache Status\n"]
        for s in statuses:
            fresh = "FRESH" if cache_mgr.is_cache_fresh(s["category"]) else "STALE"
            result.append(
                f"- **{s['category']}**: {fresh} (last: {s['last_fetched']}, source: {s['fetch_source']})"
            )

        return "\n".join(result)
    except Exception as e:
        return f"Cache status error: {e}"


# ============================================================================
# MCP RESOURCES
# ============================================================================

@mcp.resource("sota://categories")
def get_categories() -> str:
    """List all available model categories."""
    categories = {
        "image_gen": "Text/image to image generation (Z-Image-Turbo, FLUX.2, Qwen-Image)",
        "image_edit": "Image editing and inpainting (Qwen-Image-Edit, Kontext)",
        "video": "Video generation (LTX-2, Wan 2.2, HunyuanVideo)",
        "llm_local": "Local LLMs for llama.cpp (Qwen3, Llama3.3, DeepSeek-V3)",
        "llm_api": "API-based LLMs (Claude, GPT, Gemini, Grok)",
        "llm_coding": "Coding-focused LLMs (Qwen3-Coder, DeepSeek-V3, Claude)",
        "tts": "Text-to-speech (ChatterboxTTS, F5-TTS)",
        "stt": "Speech-to-text (Whisper)",
        "music": "Music generation (Suno, Udio)",
        "3d": "3D model generation (Meshy, Tripo)",
        "embeddings": "Vector embeddings (BGE, E5)"
    }

    result = ["# Model Categories\n"]
    result.append("Default: open-source models. Use open_source_only=False for all.\n")
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
