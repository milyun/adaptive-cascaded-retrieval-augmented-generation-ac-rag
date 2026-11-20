"""Shared pseudonymous visit IDs for release-only exports."""

import re
from typing import Dict, Iterable, Optional


AUDIO_KEY_RE = re.compile(
    r"(audio_\d+_\d{4}-\d{2}-\d{2}T[\d:/_\.]+[+\-]\d{4}_\d+)"
)


def extract_audio_key(value: str) -> Optional[str]:
    match = AUDIO_KEY_RE.search(value or "")
    if not match:
        return None
    return match.group(1).replace(":", "_").replace("/", "_")


def build_visit_id_map(keys: Iterable[str]) -> Dict[str, str]:
    unique_keys = sorted(set(keys))
    if not unique_keys or any(not key for key in unique_keys):
        raise ValueError("Visit keys must be non-empty")
    return {key: f"V{index:03d}" for index, key in enumerate(unique_keys, start=1)}
