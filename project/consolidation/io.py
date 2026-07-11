"""Disk adapters for the consolidation tool.

Reads/writes the on-disk mask layout produced by
project.pipeline.persistence.save_predicted_masks:

    RESULTS_DIR/masks/{image_stem}/{label}.png      # binary mask, uint8 0/255, mode "L"
    RESULTS_DIR/masks/{image_stem}/scores.json      # {"{label}.png": {"sam": ..., ...}, ...}

Uses only PIL + numpy + stdlib; no torch, no project.segmentation imports.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from project.consolidation.core import MaskRecord


def read_records_from_results(results_dir: Path) -> dict[str, list[MaskRecord]]:
    """Load all MaskRecords from a results directory's masks/ tree."""
    masks_dir = results_dir / "masks"
    if not masks_dir.exists():
        raise ValueError(f"masks directory does not exist: {masks_dir}")

    records_by_image: dict[str, list[MaskRecord]] = {}
    for image_dir in sorted(p for p in masks_dir.iterdir() if p.is_dir()):
        png_paths = sorted(image_dir.glob("*.png"))
        if not png_paths:
            # save_predicted_masks only writes scores.json when at least one
            # mask survives; an image with zero predictions has neither.
            records_by_image[image_dir.name] = []
            continue

        scores_path = image_dir / "scores.json"
        if not scores_path.exists():
            raise ValueError(f"missing scores.json for image: {image_dir}")
        with open(scores_path) as f:
            scores_payload = json.load(f)

        records = []
        for png_path in png_paths:
            filename = png_path.name
            if filename not in scores_payload:
                raise ValueError(
                    f"no score entry for mask file {filename} in {scores_path}"
                )
            mask = np.array(Image.open(png_path).convert("L")) > 127
            file_scores = scores_payload[filename]
            records.append(MaskRecord(
                label=png_path.stem,
                mask=mask,
                sam=float(file_scores["sam"]),
                scores=file_scores,
            ))
        records_by_image[image_dir.name] = records

    return records_by_image


def write_consolidated_results(
    records_by_image: dict[str, list[MaskRecord]],
    out_dir: Path,
) -> int:
    """Write consolidated records to out_dir, mirroring the input layout.

    Returns the number of mask files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for image_stem, records in records_by_image.items():
        image_dir = out_dir / image_stem
        image_dir.mkdir(parents=True, exist_ok=True)

        scores_payload = {}
        for record in records:
            filename = f"{record.label}.png"
            mask_uint8 = (record.mask.astype(np.uint8)) * 255
            Image.fromarray(mask_uint8, mode="L").save(image_dir / filename)
            scores_payload[filename] = record.scores
            n_written += 1

        with open(image_dir / "scores.json", "w") as f:
            json.dump(scores_payload, f, indent=2)

    return n_written
