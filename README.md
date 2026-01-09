# SOTA Tracker MCP Server

An MCP (Model Context Protocol) server that provides real-time SOTA (State of the Art) AI model information to Claude, preventing suggestions of outdated models.

## Problem

Claude's training data is months old, causing it to suggest outdated models:
- Redux (Oct 2024) - 15 months old
- BAGEL (May 2025) - 8 months old
- FLUX.1-dev - superseded by FLUX.2-dev

## Solution

This MCP server provides tools that Claude **automatically queries** before suggesting any AI model:

```
┌─────────────┐      query_sota("video")      ┌─────────────────┐
│   Claude    │ ─────────────────────────────►│  SOTA Tracker   │
│             │◄───────────────────────────── │  MCP Server     │
└─────────────┘  "LTX-2 (Jan 2026)"           └─────────────────┘
```

## MCP Tools

| Tool | Description | Example |
|------|-------------|---------|
| `query_sota(category)` | Get current SOTA for category | `query_sota("video")` → "LTX-2" |
| `check_freshness(model)` | Check if model is current | `check_freshness("FLUX.1-dev")` → "OUTDATED" |
| `get_forbidden()` | List all outdated models | Returns blacklist |
| `compare_models(a, b)` | Compare two models | Side-by-side comparison |
| `recent_releases(days)` | New models in past N days | Latest releases |

## Categories

- `image_gen` - Image generation (FLUX, Qwen-Image, Z-Image)
- `image_edit` - Image editing (Qwen-Image-Edit, Kontext)
- `video` - Video generation (LTX-2, Wan2.1, HunyuanVideo)
- `llm_local` - Local LLMs (Qwen2.5, Llama3.3, DeepSeek)
- `llm_api` - API LLMs (Claude, GPT, Gemini)
- `tts` - Text-to-speech (ChatterboxTTS, F5-TTS)
- `stt` - Speech-to-text (Whisper)
- `music` - Music generation (Suno, Udio)
- `3d` - 3D generation (Meshy, Tripo)
- `embeddings` - Vector embeddings (BGE, E5)

## Installation

### 1. Clone and setup

```bash
cd ~/Desktop/applications/sota-tracker-mcp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Initialize database

```bash
python init_db.py
```

### 3. Install MCP globally

```bash
claude mcp add --transport stdio --scope user sota-tracker \
  -- ~/Desktop/applications/sota-tracker-mcp/venv/bin/python \
     ~/Desktop/applications/sota-tracker-mcp/server.py
```

### 4. Restart Claude Code

The MCP server will now be available in all Claude Code sessions.

## Testing

```bash
# Test tools directly
source venv/bin/activate
python -c "
from server import _query_sota_impl, _check_freshness_impl
print(_query_sota_impl('video'))
print(_check_freshness_impl('FLUX.1-dev'))
"
```

## Updating SOTA Data

### Manual update

Edit `data/forbidden.json` for the blacklist, then run:

```bash
python init_db.py  # Re-seeds from scratch
```

### Weekly cron (optional)

```bash
# Add to crontab -e
0 3 * * 0 cd ~/Desktop/applications/sota-tracker-mcp && venv/bin/python update.py
```

## File Structure

```
sota-tracker-mcp/
├── server.py              # FastMCP server
├── init_db.py             # Database initialization
├── requirements.txt       # Dependencies
├── data/
│   ├── sota.db            # SQLite database
│   └── forbidden.json     # Human-curated blacklist
└── scrapers/              # (Future) automated scrapers
```

## Current SOTA (January 2026)

### Image Generation
1. Qwen-Image-2512 (Dec 2025)
2. Z-Image-Turbo (Nov 2025)
3. FLUX.2-dev (Nov 2025)

### Video Generation
1. LTX-2 (Jan 2026) - with audio
2. Wan 2.6 (Dec 2025) - with audio
3. HunyuanVideo 1.5 (Nov 2025)

### Local LLMs
1. Qwen2.5-72B
2. Llama3.3-70B
3. DeepSeek-V3

### Forbidden Models
- FLUX.1-dev → Use FLUX.2-dev
- Redux → Use Qwen-Image-Edit-2511
- BAGEL, OmniGen2 → Use Qwen-Image-2512
- SD 1.5/2.0/XL → Use any SOTA model
- Llama 2 → Use Llama 3.3

## License

MIT
