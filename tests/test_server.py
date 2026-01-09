"""Tests for SOTA Tracker MCP server tools."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from server import (
    _query_sota_impl,
    _check_freshness_impl,
    _get_forbidden_impl,
    _compare_models_impl,
    _recent_releases_impl,
)
from utils.classification import is_open_source


class TestQuerySota:
    """Tests for query_sota tool."""

    def test_valid_category_video(self):
        """Should return SOTA models for video category."""
        result = _query_sota_impl("video", True)
        assert "SOTA" in result
        assert "video" in result.lower()

    def test_valid_category_llm_api(self):
        """Should return SOTA models for llm_api category."""
        result = _query_sota_impl("llm_api", False)
        assert "SOTA" in result or "Models" in result

    def test_invalid_category(self):
        """Should return error for invalid category."""
        result = _query_sota_impl("invalid_category")
        assert "Invalid category" in result

    def test_open_source_only_note(self):
        """Should show note when filtering open-source."""
        result = _query_sota_impl("llm_local", True)
        assert "open-source" in result.lower()


class TestCheckFreshness:
    """Tests for check_freshness tool."""

    def test_forbidden_model(self):
        """Should identify forbidden models."""
        result = _check_freshness_impl("FLUX.1-dev")
        assert "OUTDATED" in result
        assert "FLUX.2" in result

    def test_unknown_model(self):
        """Should handle unknown models."""
        result = _check_freshness_impl("NonExistentModel12345")
        assert "UNKNOWN" in result


class TestGetForbidden:
    """Tests for get_forbidden tool."""

    def test_returns_list(self):
        """Should return forbidden models list."""
        result = _get_forbidden_impl()
        assert "Forbidden" in result
        assert "FLUX.1-dev" in result or "Redux" in result


class TestRecentReleases:
    """Tests for recent_releases tool."""

    def test_valid_days(self):
        """Should accept valid days parameter."""
        result = _recent_releases_impl(30, True)
        assert "Error" not in result

    def test_invalid_days_negative(self):
        """Should reject negative days."""
        result = _recent_releases_impl(-5, True)
        assert "Error" in result

    def test_invalid_days_too_large(self):
        """Should reject days > 365."""
        result = _recent_releases_impl(500, True)
        assert "Error" in result

    def test_invalid_days_zero(self):
        """Should reject zero days."""
        result = _recent_releases_impl(0, True)
        assert "Error" in result


class TestIsOpenSource:
    """Tests for is_open_source utility."""

    def test_closed_source_gpt(self):
        """Should identify GPT as closed-source."""
        assert is_open_source("GPT-4") is False
        assert is_open_source("gpt-4o") is False
        assert is_open_source("ChatGPT-latest") is False

    def test_closed_source_claude(self):
        """Should identify Claude as closed-source."""
        assert is_open_source("Claude Opus 4.5") is False
        assert is_open_source("claude-3-sonnet") is False

    def test_closed_source_gemini(self):
        """Should identify Gemini as closed-source."""
        assert is_open_source("Gemini 2.0 Pro") is False

    def test_open_source_llama(self):
        """Should identify Llama as open-source."""
        assert is_open_source("Llama 3.3-70B") is True
        assert is_open_source("llama-3.3-70b-instruct") is True

    def test_open_source_qwen(self):
        """Should identify Qwen as open-source."""
        assert is_open_source("Qwen 3 (Alibaba)") is True
        assert is_open_source("Qwen2.5-72B") is True

    def test_open_source_deepseek(self):
        """Should identify DeepSeek as open-source."""
        assert is_open_source("DeepSeek V3") is True


class TestCompareModels:
    """Tests for compare_models tool."""

    def test_both_unknown(self):
        """Should handle both unknown models."""
        result = _compare_models_impl("Unknown1", "Unknown2")
        assert "Neither" in result or "Not in database" in result


# Run with: python -m pytest tests/test_server.py -v
