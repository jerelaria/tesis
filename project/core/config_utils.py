"""
Configuration utilities for applying CLI overrides and saving resolved configs.

Provides dot-notation overrides (e.g., "labeler.hdbscan.min_cluster_size=20")
and automatic type inference (bool, int, float, list, str).

Usage from CLI:
    python -m main --config exp.yaml --override segmenter.score_threshold=0.7 refinement.enabled=false
"""

import json
from pathlib import Path
from typing import Any


def apply_config_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Apply dot-notation overrides to a nested config dict in-place.

    Each override has the format "key.subkey.subsubkey=value".
    Intermediate dicts are created if they do not exist.
    Values are auto-parsed via _parse_value().

    Parameters
    ----------
    cfg : dict
        The loaded YAML config dictionary.
    overrides : list[str]
        List of "KEY=VALUE" strings.

    Returns
    -------
    dict
        The same cfg dict, modified in-place (also returned for convenience).

    Raises
    ------
    ValueError
        If an override string does not contain '='.
    """
    if not overrides:
        return cfg

    print("\n  Config overrides:")
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override format: '{override}'. Expected KEY=VALUE"
            )

        key, value = override.split("=", 1)
        parts = key.split(".")

        # Navigate to the parent dict, creating intermediates as needed
        d = cfg
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                d[part] = {}
            d = d[part]

        parsed = _parse_value(value)
        d[parts[-1]] = parsed
        print(f"    {key} = {parsed!r} ({type(parsed).__name__})")

    return cfg


def save_resolved_config(
    cfg: dict,
    results_dir: Path,
    config_path: str,
    dataset_name: str,
    num_images: int,
    references_info: dict | None = None,
) -> None:
    """
    Save the fully resolved config (with overrides, resolved HDBSCAN params,
    etc.) as JSON for reproducibility.

    Parameters
    ----------
    cfg : dict
        The resolved config dictionary.
    results_dir : Path
        Experiment output directory.
    config_path : str
        Path to the original YAML config file.
    dataset_name : str
        Dataset identifier used for this run.
    num_images : int
        Number of images actually processed (after exclusions).
    references_info : dict | None
        Optional dict with few-shot reference details.
    """
    resolved = {
        "original_config": str(config_path),
        "dataset": dataset_name,
        "num_images": num_images,
        **cfg,
    }

    if references_info:
        resolved["few_shot_resolved"] = references_info

    out_path = results_dir / "resolved_config.json"
    with open(out_path, "w") as f:
        json.dump(resolved, f, indent=2, default=str)

    print(f"  Saved resolved config -> {out_path}")


def _parse_value(value: str) -> Any:
    """
    Infer Python type from a string value.

    Supports: bool, None, int, float, comma-separated list, str.
    """
    lowered = value.lower().strip()

    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none" or lowered == "null":
        return None

    # Try int
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Comma-separated list (recursive parsing for each element)
    if "," in value:
        return [_parse_value(v.strip()) for v in value.split(",")]

    # Fallback: string (strip surrounding quotes if present)
    if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
        return value[1:-1]

    return value