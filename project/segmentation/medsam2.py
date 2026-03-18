import sys
import os
import numpy as np
import torch
from typing import List
from dataclasses import dataclass

from project.core.interfaces import Segmenter
from project.core.data_types import MedicalImage, SegmentedObject
from project.segmentation.utils import (
    to_uint8,
    mask_to_bbox,
    mask_area,
    nms,
    make_point_grid,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MEDSAM2_REPO = os.path.join(_THIS_DIR, "..", "..", "weights", "MedSAM2")
if _MEDSAM2_REPO not in sys.path:
    sys.path.insert(0, os.path.abspath(_MEDSAM2_REPO))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ---------------------------------------------------------------------------
# Configuración por defecto (se puede sobreescribir desde YAML)
# ---------------------------------------------------------------------------

_CKPT_PATH     = "weights/MedSAM2/checkpoints/MedSAM2_latest.pt" # Path to the checkpoint, relative to the project root
_HYDRA_CFG     = "configs/sam2.1_hiera_t512.yaml" # Path to the config file, relative to the MedSAM2 repo


@dataclass
class MedSAM2Config:
    device: str = "cuda"
    grid_side: int = 6
    score_threshold: float = 0.80
    iou_threshold: float = 0.50

class MedSAM2Segmenter(Segmenter):
    """
    Segment a MedicalImage using MedSAM2 with a grid of points.
    
    Parameters
    ----------
    config : MedSAM2Config
        Hyperparameters for the segmenter.
    """

    def __init__(self, config: MedSAM2Config = MedSAM2Config()):
        self.config = config
        if config.device != "cpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif config.device != "cpu" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self._predictor = self._build_predictor()

    # ------------------------------------------------------------------
    # Constructor alternativo
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, yaml_path: str) -> "MedSAM2Segmenter":
        """Carga hiperparámetros desde un YAML del proyecto."""
        import yaml
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        config = MedSAM2Config(**raw.get("segmenter", {}))
        return cls(config)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def segment(self, image: MedicalImage) -> List[SegmentedObject]:
        """
        Receives a MedicalImage (RGB float32, HxWx3) and returns
        List[SegmentedObject] with mask, bbox and confidence.
        """
        img_uint8 = to_uint8(image.volume)
        self._predictor.set_image(img_uint8)

        h, w  = img_uint8.shape[:2]
        points = make_point_grid(h, w, self.config.grid_side)

        raw      = self._predict_all_points(points)

        # print("Segmented masks:", len(raw))
        # sorted_raw = sorted(raw, key=lambda x: x[1], reverse=True)
        # print("Top 3 masks by score:")
        # for i, (mask, score) in enumerate(sorted_raw[:3]):
        #     print(f"  {i}: score={score:.2f}")
            
        filtered = [
            (mask, score) for mask, score in raw
            if score >= self.config.score_threshold
        ]
        kept = nms(filtered, iou_threshold=self.config.iou_threshold)

        objects = []
        for idx, (mask, score) in enumerate(kept):
            objects.append(SegmentedObject(
                mask=mask,
                source_image=image,
                confidence=float(score),
            ))

        return objects

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _build_predictor(self) -> SAM2ImagePredictor:
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        ckpt = os.path.join(project_root, _CKPT_PATH)
        
        # Hydra necesita que el cwd sea la carpeta raíz del repo MedSAM2
        medsam2_root = os.path.join(project_root, "weights", "MedSAM2")
        original_cwd = os.getcwd()
        os.chdir(medsam2_root)
        
        try:
            model = build_sam2(_HYDRA_CFG, ckpt, device=self.device)
        finally:
            os.chdir(original_cwd)  # siempre restaurar el cwd original
        
        return SAM2ImagePredictor(model)

    def _predict_all_points(self, points: np.ndarray) -> list:
        """
        Por cada punto en la grilla llama al predictor y conserva
        la máscara con mayor score.
        Devuelve lista de (mask bool H×W, score float).
        """
        label   = np.array([1], dtype=np.int32)  # punto positivo
        results = []

        for pt in points:
            masks, scores, _ = self._predictor.predict(
                point_coords=pt[np.newaxis, :],
                point_labels=label,
                multimask_output=True,
            )
            best = int(np.argmax(scores))
            results.append((masks[best].astype(bool), float(scores[best])))

        return results
