import logging
import os
import shutil
import tempfile
from contextlib import contextmanager

import numpy as np

from project.core.data_types import MedicalImage, SegmentedObject

logger = logging.getLogger(__name__)


@contextmanager
def _video_session(video_pred, ref_uint8s, target_uint8s, mask_register_fn):
    """
    Create tmp_dir, write frames, init state, register masks.

    ref_uint8s    : list of uint8 arrays written as frames 0..K-1
    target_uint8s : list of uint8 arrays written as frames K..K+N-1
    mask_register_fn : callable(video_pred, state) -> registration_result

    Yields (state, registration_result).
    Resets state and removes tmp_dir on exit.
    """
    from PIL import Image as PILImage

    K = len(ref_uint8s)
    tmp_dir = tempfile.mkdtemp(prefix="medsam2_video_")
    state = None
    try:
        for i, uint8 in enumerate(ref_uint8s):
            PILImage.fromarray(uint8).save(os.path.join(tmp_dir, f"{i:05d}.jpg"))
        for j, uint8 in enumerate(target_uint8s):
            PILImage.fromarray(uint8).save(
                os.path.join(tmp_dir, f"{K + j:05d}.jpg")
            )
        state = video_pred.init_state(video_path=tmp_dir)
        reg_result = mask_register_fn(video_pred, state)
        yield state, reg_result
    finally:
        if state is not None:
            video_pred.reset_state(state)
        shutil.rmtree(tmp_dir)


def register_multi_frame_masks(
    video_pred, state, references: list
) -> dict[int, str]:
    """
    Register organ masks from K FewShotReference objects on frames 0..K-1.

    Each organ gets a unique obj_id (consistent across frames). Returns the
    obj_id -> organ_name mapping used when collecting propagation results.
    """
    all_organ_names: list[str] = []
    for ref in references:
        for organ_name in ref.masks:
            if organ_name not in all_organ_names:
                all_organ_names.append(organ_name)

    organ_to_obj_id = {name: idx + 1 for idx, name in enumerate(all_organ_names)}
    obj_id_to_organ = {v: k for k, v in organ_to_obj_id.items()}

    for frame_idx, ref in enumerate(references):
        for organ_name, mask in ref.masks.items():
            obj_id = organ_to_obj_id[organ_name]
            video_pred.add_new_mask(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=mask.astype(np.float32),
            )

    logger.info(
        f"[VIDEO] Registered {len(all_organ_names)} organs "
        f"across {len(references)} reference frames: "
        f"{', '.join(all_organ_names)}"
    )
    return obj_id_to_organ


def collect_frame_results(
    video_res_masks,
    obj_ids: list,
    organ_names_by_obj_id: dict[int, str],
    source_image: MedicalImage,
) -> list[SegmentedObject]:
    """Convert video predictor output for one frame into SegmentedObjects."""
    objects = []
    for pos, obj_id in enumerate(obj_ids):
        logits = video_res_masks[pos].cpu().numpy().squeeze()
        binary_mask = logits > 0.0

        if not binary_mask.any():
            organ = organ_names_by_obj_id.get(obj_id, "unknown")
            logger.debug(f"[VIDEO] Empty mask for '{organ}', skipping")
            continue

        objects.append(SegmentedObject(
            mask=binary_mask,
            source_image=source_image,
            confidence=logits_to_confidence(logits, binary_mask),
            label=organ_names_by_obj_id.get(obj_id, "unknown"),
        ))

    return objects


def logits_to_confidence(logits: np.ndarray, binary_mask: np.ndarray) -> float:
    fg_logits = logits[binary_mask]
    if len(fg_logits) == 0:
        return 0.0
    sigmoid_vals = 1.0 / (1.0 + np.exp(-np.clip(fg_logits, -20, 20)))
    return float(sigmoid_vals.mean())
