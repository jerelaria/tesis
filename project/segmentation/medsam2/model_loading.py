import os
import sys
import logging
from contextlib import chdir

logger = logging.getLogger(__name__)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))

_MEDSAM2_REPO = os.path.join(_PROJECT_ROOT, "weights", "MedSAM2")
if _MEDSAM2_REPO not in sys.path:
    sys.path.insert(0, _MEDSAM2_REPO)

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

_CKPT_PATH = os.path.join(_PROJECT_ROOT, "weights", "MedSAM2", "checkpoints", "MedSAM2_latest.pt")
_HYDRA_CFG = "configs/sam2.1_hiera_t512.yaml"


def build_image_predictor(device) -> SAM2ImagePredictor:
    with chdir(_MEDSAM2_REPO):
        model = build_sam2(_HYDRA_CFG, _CKPT_PATH, device=device)
    return SAM2ImagePredictor(model)


def build_video_predictor(device):
    with chdir(_MEDSAM2_REPO):
        predictor = build_sam2_video_predictor(_HYDRA_CFG, _CKPT_PATH, device=device)
    logger.info("[MedSAM2] Video predictor loaded.")
    return predictor
