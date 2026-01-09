"""
Artificial Analysis scraper.

Scrapes leaderboard data from artificialanalysis.ai.
No public API available, so we parse their HTML/JSON.

Sources:
- https://artificialanalysis.ai/leaderboards/models (LLMs)
- https://artificialanalysis.ai/text-to-image/arena (Image Gen)
- https://artificialanalysis.ai/text-to-video/arena (Video Gen)
"""

import json
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional
from html.parser import HTMLParser

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.classification import is_open_source


class ArtificialAnalysisFetcher:
    """Scrape SOTA data from Artificial Analysis."""

    BASE_URL = "https://artificialanalysis.ai"

    ENDPOINTS = {
        "llm": "/leaderboards/models",
        "image_gen": "/text-to-image/arena",
        "video": "/text-to-video/arena",
        "tts": "/speech/arena",
    }

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def fetch_category(self, category: str) -> list[dict]:
        """
        Fetch leaderboard for a category.

        Args:
            category: One of 'llm', 'image_gen', 'video', 'tts'
        """
        endpoint = self.ENDPOINTS.get(category)
        if not endpoint:
            print(f"Unknown category: {category}")
            return []

        url = f"{self.BASE_URL}{endpoint}"

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; sota-tracker-mcp/1.0)",
                    "Accept": "text/html,application/xhtml+xml",
                }
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                html = resp.read().decode()
                return self._parse_html(html, category)
        except urllib.error.HTTPError as e:
            print(f"Artificial Analysis fetch failed ({category}): {e.code}")
            return []
        except Exception as e:
            print(f"Artificial Analysis error ({category}): {e}")
            return []

    def _parse_html(self, html: str, category: str) -> list[dict]:
        """
        Parse HTML to extract leaderboard data.

        Artificial Analysis embeds data in Next.js JSON or script tags.
        """
        models = []

        # Try to find embedded JSON data (Next.js pattern)
        json_match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
        if json_match:
            try:
                next_data = json.loads(json_match.group(1))
                models = self._parse_next_data(next_data, category)
                if models:
                    return models
            except json.JSONDecodeError:
                pass

        # Fallback: Parse table structure
        models = self._parse_table(html, category)

        return models

    def _parse_next_data(self, data: dict, category: str) -> list[dict]:
        """Extract models from Next.js embedded data."""
        models = []

        # Navigate nested structure - this varies by page
        try:
            props = data.get("props", {}).get("pageProps", {})
            leaderboard = props.get("leaderboard", props.get("models", props.get("data", [])))

            if isinstance(leaderboard, list):
                for rank, entry in enumerate(leaderboard[:20], 1):
                    model = self._entry_to_model(entry, category, rank)
                    if model:
                        models.append(model)
        except Exception as e:
            print(f"Error parsing Next.js data: {e}")

        return models

    def _parse_table(self, html: str, category: str) -> list[dict]:
        """Fallback: Parse HTML table rows."""
        models = []

        # Simple regex to find model names and scores
        # Pattern varies by page, this is a best-effort fallback
        patterns = [
            r'<td[^>]*>(\d+)</td>\s*<td[^>]*>([^<]+)</td>\s*<td[^>]*>(\d+\.?\d*)</td>',
            r'"name":\s*"([^"]+)".*?"elo":\s*(\d+)',
            r'"model":\s*"([^"]+)".*?"score":\s*(\d+\.?\d*)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, html, re.DOTALL)
            if matches:
                for rank, match in enumerate(matches[:20], 1):
                    if len(match) >= 2:
                        name = match[1] if len(match) == 3 else match[0]
                        score = match[2] if len(match) == 3 else match[1]
                        models.append({
                            "id": name.lower().replace(" ", "-").replace("/", "-"),
                            "name": name,
                            "category": self._map_category(category),
                            "is_open_source": is_open_source(name),
                            "sota_rank": rank,
                            "metrics": {
                                "elo": float(score) if score else None,
                                "notes": f"Artificial Analysis rank #{rank}",
                                "source": "artificial_analysis"
                            }
                        })
                break

        return models

    def _entry_to_model(self, entry: dict, category: str, rank: int) -> Optional[dict]:
        """Convert a leaderboard entry to our model format."""
        name = entry.get("name", entry.get("model", entry.get("model_name")))
        if not name:
            return None

        elo = entry.get("elo", entry.get("score", entry.get("rating")))

        return {
            "id": name.lower().replace(" ", "-").replace("/", "-"),
            "name": name,
            "category": self._map_category(category),
            "is_open_source": entry.get("is_open_source", is_open_source(name)),
            "sota_rank": rank,
            "metrics": {
                "elo": elo,
                "notes": entry.get("description", f"Artificial Analysis #{rank}"),
                "price_input": entry.get("price_input"),
                "price_output": entry.get("price_output"),
                "speed": entry.get("speed"),
                "source": "artificial_analysis"
            }
        }

    def _map_category(self, aa_category: str) -> str:
        """Map Artificial Analysis category to our categories."""
        mapping = {
            "llm": "llm_api",
            "image_gen": "image_gen",
            "video": "video",
            "tts": "tts",
        }
        return mapping.get(aa_category, aa_category)



# Quick test
if __name__ == "__main__":
    fetcher = ArtificialAnalysisFetcher()

    print("=== Artificial Analysis LLM Leaderboard ===")
    llms = fetcher.fetch_category("llm")
    for m in llms[:5]:
        elo = m["metrics"].get("elo", "N/A")
        badge = "" if m["is_open_source"] else " [CLOSED]"
        print(f"  #{m['sota_rank']} {m['name']}{badge}: Elo {elo}")
