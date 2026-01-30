# SOTA Tracker MCP

MCP server providing real-time AI model rankings to Claude, preventing outdated model suggestions.

## Architecture: Data Distribution Pattern

**This repo uses git as a data distribution mechanism.**

```
GitHub Action (daily at 6 AM UTC)
    → Scrapes LMArena, HuggingFace, Civitai, Artificial Analysis
    → Updates data/sota.db
    → Commits & pushes to GitHub
    → Weekly releases on Sundays
```

### What IS Tracked (Intentionally)

| File | Why |
|------|-----|
| `data/sota.db` | Distribution mechanism - users pull for fresh data |
| `data/*_latest.json` | Raw scrape results |
| `data/sota_export.*` | Exports for non-sqlite consumers |
| `data/forbidden.json` | Curated blocklist |

### What Should Be Gitignored

| File | Why |
|------|-----|
| `data/hardware_profiles.json` | User-specific (GPU specs differ per machine) |

### Git Workflow

**Before pushing changes:**
```bash
git pull --rebase  # Get latest scraped data first
```

Conflicts with `sota.db` mean GitHub Action pushed newer data. Take remote version:
```bash
git checkout --theirs data/sota.db
```

---

## Key Files

| File | Purpose |
|------|---------|
| `init_db.py` | Seeds database with curated SOTA data |
| `scrapers/run_all.py` | Runs all scrapers (used by GitHub Action) |
| `server.py` | MCP server entry point |
| `.github/workflows/daily-scrape.yml` | Automation config |

## Adding New Models

1. Edit `init_db.py` - add to `seed_sota_models()`
2. Run `python init_db.py` to rebuild local db
3. Commit and push - GitHub Action will merge with scraped data

## Model Constraints

When adding models with specific workflow settings (like Qwen's shift parameter), add a `constraints` field:

```python
"metrics": {
    "constraints": {
        "cfg": {"min": 3.0, "max": 5.0, "default": 3.5},
        "shift": {"min": 7.0, "max": 13.0, "default": 7.0, "note": "CRITICAL info"}
    }
}
```
