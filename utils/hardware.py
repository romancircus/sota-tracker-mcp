"""Hardware profile utilities for personalized model recommendations."""

import json
import re
import socket
from pathlib import Path
from typing import Any, Optional

# Path to hardware profiles config
DATA_DIR = Path(__file__).parent.parent / "data"
HARDWARE_PROFILES_PATH = DATA_DIR / "hardware_profiles.json"

# Default profile template
DEFAULT_PROFILE = {
    "gpu": "Unknown",
    "vram_gb": 8,
    "ram_gb": 16,
    "cpu_threads": 4,
    "preferences": {
        "uncensored": False,
        "local_first": True,
        "cost_sensitive": True
    }
}


def load_hardware_profiles() -> dict:
    """Load all hardware profiles from config file."""
    if not HARDWARE_PROFILES_PATH.exists():
        return {"current": None, "profiles": {}}

    with open(HARDWARE_PROFILES_PATH, "r") as f:
        return json.load(f)


def save_hardware_profiles(profiles: dict) -> None:
    """Save hardware profiles to config file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HARDWARE_PROFILES_PATH, "w") as f:
        json.dump(profiles, f, indent=2)


def get_current_profile() -> Optional[dict]:
    """Get the currently active hardware profile."""
    profiles = load_hardware_profiles()
    current_name = profiles.get("current")

    if not current_name:
        return None

    return profiles.get("profiles", {}).get(current_name)


def get_profile_with_defaults() -> dict:
    """Get current profile, falling back to defaults if not configured."""
    profile = get_current_profile()
    if profile:
        # Merge with defaults to ensure all keys exist
        result = DEFAULT_PROFILE.copy()
        result.update(profile)
        if "preferences" in profile:
            result["preferences"] = {**DEFAULT_PROFILE["preferences"], **profile["preferences"]}
        return result
    return DEFAULT_PROFILE.copy()


def configure_profile(
    profile_name: Optional[str] = None,
    vram_gb: Optional[int] = None,
    gpu: Optional[str] = None,
    ram_gb: Optional[int] = None,
    cpu_threads: Optional[int] = None,
    uncensored_preference: Optional[bool] = None,
    local_first: Optional[bool] = None,
    cost_sensitive: Optional[bool] = None
) -> dict:
    """Configure or update a hardware profile."""
    profiles = load_hardware_profiles()

    # Use hostname if no profile name provided
    if not profile_name:
        profile_name = socket.gethostname()

    # Get existing profile or create new one
    existing = profiles.get("profiles", {}).get(profile_name, DEFAULT_PROFILE.copy())

    # Update fields if provided
    if vram_gb is not None:
        existing["vram_gb"] = vram_gb
    if gpu is not None:
        existing["gpu"] = gpu
    if ram_gb is not None:
        existing["ram_gb"] = ram_gb
    if cpu_threads is not None:
        existing["cpu_threads"] = cpu_threads

    # Update preferences
    if "preferences" not in existing:
        existing["preferences"] = DEFAULT_PROFILE["preferences"].copy()
    if uncensored_preference is not None:
        existing["preferences"]["uncensored"] = uncensored_preference
    if local_first is not None:
        existing["preferences"]["local_first"] = local_first
    if cost_sensitive is not None:
        existing["preferences"]["cost_sensitive"] = cost_sensitive

    # Save profile
    if "profiles" not in profiles:
        profiles["profiles"] = {}
    profiles["profiles"][profile_name] = existing
    profiles["current"] = profile_name

    save_hardware_profiles(profiles)

    return {"profile_name": profile_name, **existing}


def parse_vram_string(vram_str: str) -> Optional[int]:
    """Parse VRAM string like '16GB' or '24 GB GGUF' to integer GB."""
    if not vram_str:
        return None

    # Extract number followed by GB
    match = re.search(r'(\d+)\s*(?:GB|gb)', str(vram_str))
    if match:
        return int(match.group(1))

    # Try plain integer
    try:
        return int(vram_str)
    except (ValueError, TypeError):
        return None


def vram_fits(model_vram: Any, available_vram_gb: int) -> bool:
    """Check if a model fits in available VRAM."""
    if model_vram is None:
        # If no VRAM info, assume it fits (be permissive)
        return True

    required = parse_vram_string(str(model_vram))
    if required is None:
        return True

    return required <= available_vram_gb


def get_available_vram(concurrent_usage_gb: int = 0) -> int:
    """Calculate available VRAM based on profile and concurrent usage."""
    profile = get_profile_with_defaults()
    total_vram = profile.get("vram_gb", 8)
    available = total_vram - concurrent_usage_gb
    return max(0, available)


def get_concurrent_vram_estimate(workload: str) -> int:
    """Estimate VRAM usage for common concurrent workloads."""
    estimates = {
        "image_gen": 24,      # FLUX.2-dev, Qwen-Image
        "video_gen": 24,      # HunyuanVideo, Wan2.2
        "image_edit": 18,     # FLUX.1-Kontext
        "stable_diffusion": 12,
        "comfyui": 16,
        "blender": 8,
        "gaming": 12,
        "desktop": 2,         # Normal desktop compositing
        "none": 0
    }
    return estimates.get(workload.lower(), 0) if workload else 0
