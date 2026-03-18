import sys
import os
import numpy as np
import torch
from typing import List
from dataclasses import dataclass, field

from project.core.interfaces import Segmenter
from project.core.data_types import MedicalImage, SegmentedObject

# ---------------------------------------------------------------------------
# Add MedSAM3 repo to path so its internal modules are importable
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MEDSAM3_REPO = os.path.join(_THIS_DIR, "..", "..", "weights", "MedSAM3")
if _MEDSAM3_REPO not in sys.path:
    sys.path.insert(0, os.path.abspath(_MEDSAM3_REPO))

from infer_sam import SAM3LoRAInference  # MedSAM3's inference class

# ---------------------------------------------------------------------------
# Default paths (relative to project root)
# ---------------------------------------------------------------------------

_LORA_WEIGHTS = "weights/MedSAM3/checkpoints/best_lora_weights.pt"
_CONFIG_PATH  = "weights/MedSAM3/configs/full_lora_config.yaml"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class MedSAM3Config:
    device: str = "cuda"
    prompts: List[str] = field(default_factory=lambda: ["organ"])  # text prompts
    score_threshold: float = 0.50
    nms_iou: float = 0.50
    resolution: int = 1008
    lora_weights: str = _LORA_WEIGHTS
    config_path: str = _CONFIG_PATH


# ---------------------------------------------------------------------------
# Main segmenter class
# ---------------------------------------------------------------------------

class MedSAM3Segmenter(Segmenter):
    """
    Segment a MedicalImage using MedSAM3 (SAM3 + LoRA fine-tuning).

    MedSAM3 is text-prompted: you pass one or more natural language descriptions
    (e.g. "liver", "lung tumor") and the model returns masks for each concept.

    Unlike MedSAM2, there is no point grid — SAM3's architecture uses a text
    encoder as the query mechanism, so point prompts are not supported.

    Parameters
    ----------
    config : MedSAM3Config
        Hyperparameters and paths for the segmenter.
    """

    def __init__(self, config: MedSAM3Config = MedSAM3Config()):
        self.config = config
        self._inferencer = self._build_inferencer()

    # ------------------------------------------------------------------
    # Alternative constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, yaml_path: str) -> "MedSAM3Segmenter":
        """Load hyperparameters from a project YAML file."""
        import yaml
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        cfg = raw.get("segmenter", {})
        config = MedSAM3Config(**cfg)
        return cls(config)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def segment(self, image: MedicalImage) -> List[SegmentedObject]:
        """
        Receives a MedicalImage (RGB float32, H x W x 3) and returns
        List[SegmentedObject] with mask and confidence.

        Each text prompt in self.config.prompts may generate multiple detections.
        All detections from all prompts are returned as separate SegmentedObjects.

        The real output format from SAM3LoRAInference.predict() is:
        {
            0: {'prompt': str, 'boxes': ndarray, 'scores': ndarray,
                'masks': ndarray (N, H, W), 'num_detections': int},
            1: {...},
            '_image': PIL.Image   # ignored here
        }
        """
        tmp_path = "/tmp/_medsam3_input.png"
        self._save_image(image.volume, tmp_path)

        # predict() processes all prompts in a single call
        raw_results = self._inferencer.predict(tmp_path, self.config.prompts)

        objects = []
        obj_id = 0

        # Iterate over prompt indices (skip the '_image' key)
        for key in sorted(k for k in raw_results if k != "_image"):
            result = raw_results[key]

            if result["num_detections"] == 0:
                continue

            masks  = result["masks"]   # np.ndarray (N, H, W) bool
            scores = result["scores"]  # np.ndarray (N,) float

            for i in range(result["num_detections"]):
                objects.append(SegmentedObject(
                    id=obj_id,
                    mask=masks[i].astype(bool),
                    source_image=image,
                    confidence=float(scores[i]),
                    label=result["prompt"],
                ))
                obj_id += 1

        return objects

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _build_inferencer(self) -> SAM3LoRAInference:
        """
        Instantiate SAM3LoRAInference with correct absolute paths.
        MedSAM3's inference class handles loading SAM3 base + applying LoRA.
        """
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        lora_weights = os.path.join(project_root, self.config.lora_weights)
        config_path  = os.path.join(project_root, self.config.config_path)

        # MedSAM3 needs its own repo as cwd to resolve internal imports
        medsam3_root = os.path.join(project_root, "weights", "MedSAM3")
        original_cwd = os.getcwd()
        os.chdir(medsam3_root)

        try:
            inferencer = SAM3LoRAInference(
                config_path=config_path,
                weights_path=lora_weights,
                resolution=self.config.resolution,
                detection_threshold=self.config.score_threshold,
                nms_iou_threshold=self.config.nms_iou,
                device=self.config.device,
            )
        finally:
            os.chdir(original_cwd)  # always restore original cwd

        return inferencer

    def _save_image(self, volume: np.ndarray, path: str) -> None:
        """
        Save a float32 H x W x 3 array as PNG for MedSAM3's file-based API.
        Normalizes to [0, 255] uint8 if needed.
        """
        from PIL import Image as PILImage

        if volume.dtype != np.uint8:
            v_min, v_max = volume.min(), volume.max()
            if v_max > v_min:
                volume = (volume - v_min) / (v_max - v_min) * 255.0
            volume = volume.clip(0, 255).astype(np.uint8)

        PILImage.fromarray(volume).save(path)