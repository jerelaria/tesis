import sys
import os
import tempfile
import shutil
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
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

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

_CKPT_PATH = "weights/MedSAM2/checkpoints/MedSAM2_latest.pt"
_HYDRA_CFG = "configs/sam2.1_hiera_t512.yaml"


@dataclass
class MedSAM2Config:
    device: str = "cuda"
    grid_side: int = 6
    score_threshold: float = 0.80
    iou_threshold: float = 0.50


class MedSAM2Segmenter(Segmenter):
    """
    Segment a MedicalImage using MedSAM2.

    Four public methods:

    1. segment() — grid of points, image predictor. Unsupervised mode only.

    2. segment_with_video_prompts() — (K+1)-frame video: K reference images
       (each with multiple organ masks) + 1 target. Independent mode.

    3. segment_batch_iterative() — (K+N)-frame video: K references + N targets.
       Memory accumulates from references AND previously segmented targets.

    4. segment_with_multi_reference() — (K+1)-frame video: K references each
       showing ONE organ + 1 target. Used exclusively by refinement.

    In methods 2 and 3, each reference image can have multiple organ masks
    (registered as separate obj_ids on the same frame). SAM2 propagates all
    organs simultaneously.

    The video predictor is lazy-loaded on first use.
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
        self._video_predictor = None

    @classmethod
    def from_config(cls, yaml_path: str) -> "MedSAM2Segmenter":
        import yaml
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        config = MedSAM2Config(**raw.get("segmenter", {}))
        return cls(config)

    # ------------------------------------------------------------------
    # Public: encode image for embedding extraction
    # ------------------------------------------------------------------

    def encode_image(self, image: MedicalImage) -> torch.Tensor:
        """
        Run an image through the Hiera encoder and return the embedding.

        Uses the image predictor (same backbone weights as the video
        predictor). Safe to call after any segmentation method -- the
        video predictor state is not affected.

        Parameters
        ----------
        image : MedicalImage
            Image to encode.

        Returns
        -------
        torch.Tensor
            Image embedding of shape (1, 256, 64, 64).
        """
        img_uint8 = to_uint8(image.volume)
        self._predictor.set_image(img_uint8)
        return self._predictor._features["image_embed"]

    # ------------------------------------------------------------------
    # 1. Grid-based segmentation (unsupervised only)
    # ------------------------------------------------------------------

    def segment(self, image: MedicalImage) -> List[SegmentedObject]:
        """Grid of point prompts -> image predictor -> SegmentedObjects."""
        img_uint8 = to_uint8(image.volume)
        self._predictor.set_image(img_uint8)

        h, w = img_uint8.shape[:2]
        points = make_point_grid(h, w, self.config.grid_side)
        raw = self._predict_all_points(points)

        filtered = [
            (mask, score) for mask, score in raw
            if score >= self.config.score_threshold
        ]
        kept = nms(filtered, iou_threshold=self.config.iou_threshold)

        return [
            SegmentedObject(mask=mask, source_image=image, confidence=float(score))
            for mask, score in kept
        ]

    # ------------------------------------------------------------------
    # 2. INDEPENDENT: K refs + 1 target per call (few-shot independent)
    # ------------------------------------------------------------------

    def segment_with_video_prompts(
        self,
        target_image: MedicalImage,
        references: list,
    ) -> list[SegmentedObject]:
        """
        Propagate masks from K reference images to one target.

        Builds a (K+1)-frame video:
        - Frames 0..K-1: reference images (each with its organ masks)
        - Frame K: target image

        SAM2 sees all K references before propagating to the target.
        Each call is independent (no state between calls). Order-invariant
        with respect to targets (but reference order may matter slightly).

        Parameters
        ----------
        target_image : MedicalImage
            The image to segment.
        references : list[FewShotReference]
            Reference images with organ masks. Each reference has:
            - .volume: np.ndarray (H, W, 3)
            - .masks: dict[str, np.ndarray] organ_name -> (H, W) bool

        Returns
        -------
        list[SegmentedObject]
            One per successfully propagated organ.
            Each has label = organ_name.
        """
        from PIL import Image as PILImage

        if not references:
            return []

        video_pred = self._get_video_predictor()
        K = len(references)
        target_frame_idx = K

        tmp_dir = tempfile.mkdtemp(prefix="medsam2_video_")
        try:
            # Frames 0..K-1: reference images
            for i, ref in enumerate(references):
                ref_uint8 = to_uint8(ref.volume)
                PILImage.fromarray(ref_uint8).save(
                    os.path.join(tmp_dir, f"{i:05d}.jpg")
                )

            # Frame K: target
            tgt_uint8 = to_uint8(target_image.volume)
            PILImage.fromarray(tgt_uint8).save(
                os.path.join(tmp_dir, f"{target_frame_idx:05d}.jpg")
            )

            state = video_pred.init_state(video_path=tmp_dir)

            # Register organ masks on each reference frame
            organ_names_by_obj_id = self._register_multi_frame_masks(
                video_pred, state, references
            )

            # Propagate and collect only the target frame
            objects = []
            for frame_idx, obj_ids, video_res_masks in (
                video_pred.propagate_in_video(state)
            ):
                if frame_idx != target_frame_idx:
                    continue
                objects.extend(
                    self._collect_frame_results(
                        video_res_masks, obj_ids, organ_names_by_obj_id,
                        target_image,
                    )
                )

            video_pred.reset_state(state)

        finally:
            shutil.rmtree(tmp_dir)

        return objects

    # ------------------------------------------------------------------
    # 3. ITERATIVE: K refs + N targets, memory accumulates
    # ------------------------------------------------------------------

    def segment_batch_iterative(
        self,
        target_entries: list[tuple[Path, MedicalImage]],
        references: list,
    ) -> dict[Path, list[SegmentedObject]]:
        """
        Propagate masks from K references to N targets in a single pass.

        Builds a (K+N)-frame video:
        - Frames 0..K-1: reference images (with organ masks)
        - Frames K..K+N-1: target images

        SAM2 memory accumulates: target at frame K+i sees all K references
        AND all previously processed targets (frames K..K+i-1).

        Trade-offs vs independent:
        - Pro: later targets benefit from accumulated evidence
        - Con: order-dependent, errors propagate forward

        Parameters
        ----------
        target_entries : list[tuple[Path, MedicalImage]]
            Ordered (path, image) pairs for target images.
        references : list[FewShotReference]
            Reference images with organ masks.

        Returns
        -------
        dict[Path, list[SegmentedObject]]
            Per-image propagated objects.
        """
        from PIL import Image as PILImage

        if not target_entries or not references:
            return {}

        video_pred = self._get_video_predictor()
        K = len(references)

        frame_to_path: dict[int, Path] = {}
        frame_to_image: dict[int, MedicalImage] = {}

        tmp_dir = tempfile.mkdtemp(prefix="medsam2_batch_")
        try:
            # Frames 0..K-1: references
            for i, ref in enumerate(references):
                ref_uint8 = to_uint8(ref.volume)
                PILImage.fromarray(ref_uint8).save(
                    os.path.join(tmp_dir, f"{i:05d}.jpg")
                )

            # Frames K..K+N-1: targets
            for i, (path, med_image) in enumerate(target_entries):
                frame_idx = K + i
                frame_to_path[frame_idx] = path
                frame_to_image[frame_idx] = med_image

                tgt_uint8 = to_uint8(med_image.volume)
                PILImage.fromarray(tgt_uint8).save(
                    os.path.join(tmp_dir, f"{frame_idx:05d}.jpg")
                )

            total_frames = K + len(target_entries)
            print(f"    [VIDEO BATCH] {total_frames}-frame video "
                  f"({K} refs + {len(target_entries)} targets)")

            state = video_pred.init_state(video_path=tmp_dir)

            organ_names_by_obj_id = self._register_multi_frame_masks(
                video_pred, state, references
            )

            results: dict[Path, list[SegmentedObject]] = {
                path: [] for path, _ in target_entries
            }

            propagated_count = 0
            for frame_idx, obj_ids, video_res_masks in (
                video_pred.propagate_in_video(state)
            ):
                if frame_idx not in frame_to_path:
                    continue

                path = frame_to_path[frame_idx]
                source_image = frame_to_image[frame_idx]

                frame_objects = self._collect_frame_results(
                    video_res_masks, obj_ids, organ_names_by_obj_id,
                    source_image,
                )
                results[path].extend(frame_objects)
                propagated_count += len(frame_objects)

                target_idx = frame_idx - K
                n_targets = len(target_entries)
                if target_idx % 50 == 0 or target_idx == n_targets - 1:
                    print(f"    [VIDEO BATCH] Target {target_idx+1}/{n_targets}")

            video_pred.reset_state(state)
            print(f"    [VIDEO BATCH] {propagated_count} objects "
                  f"across {len(target_entries)} images")

        finally:
            shutil.rmtree(tmp_dir)

        return results

    # ------------------------------------------------------------------
    # 4. MULTI-REFERENCE: K refs (single organ each) -> 1 target
    #    Used exclusively by refinement.
    # ------------------------------------------------------------------

    def segment_with_multi_reference(
        self,
        target_image: MedicalImage,
        reference_entries: list[tuple[np.ndarray, np.ndarray]],
        organ_name: str,
    ) -> SegmentedObject | None:
        """
        Propagate one organ from K reference images to one target.

        Frame 0..K-1: reference images (each with one mask for the same organ)
        Frame K: target image

        All K masks use obj_id=1 (same organ). SAM2 sees K examples of the
        organ in different patients before propagating to the target.

        Used exclusively by retroactive refinement.

        Parameters
        ----------
        target_image : MedicalImage
            Image where the organ is missing.
        reference_entries : list[tuple[np.ndarray, np.ndarray]]
            List of (volume, mask) pairs. Each mask is for the same organ.
        organ_name : str
            Name of the organ being recovered.

        Returns
        -------
        SegmentedObject or None
            Recovered object, or None if propagation failed.
        """
        from PIL import Image as PILImage

        if not reference_entries:
            return None

        video_pred = self._get_video_predictor()

        tmp_dir = tempfile.mkdtemp(prefix="medsam2_multiref_")
        try:
            for i, (ref_vol, _) in enumerate(reference_entries):
                ref_uint8 = to_uint8(ref_vol)
                PILImage.fromarray(ref_uint8).save(
                    os.path.join(tmp_dir, f"{i:05d}.jpg")
                )

            target_frame_idx = len(reference_entries)
            tgt_uint8 = to_uint8(target_image.volume)
            PILImage.fromarray(tgt_uint8).save(
                os.path.join(tmp_dir, f"{target_frame_idx:05d}.jpg")
            )

            state = video_pred.init_state(video_path=tmp_dir)

            # All masks use obj_id=1 (same organ, different patients)
            for i, (_, ref_mask) in enumerate(reference_entries):
                video_pred.add_new_mask(
                    inference_state=state,
                    frame_idx=i,
                    obj_id=1,
                    mask=ref_mask.astype(np.float32),
                )

            result = None
            for frame_idx, obj_ids, video_res_masks in (
                video_pred.propagate_in_video(state)
            ):
                if frame_idx != target_frame_idx:
                    continue
                if len(video_res_masks) == 0:
                    continue

                logits = video_res_masks[0].cpu().numpy().squeeze()
                binary_mask = logits > 0.0

                if not binary_mask.any():
                    print(f"    [MULTI-REF] Empty mask for '{organ_name}'")
                    break

                result = SegmentedObject(
                    mask=binary_mask,
                    source_image=target_image,
                    confidence=self._logits_to_confidence(logits, binary_mask),
                    label=organ_name,
                )
                break

            video_pred.reset_state(state)

        finally:
            shutil.rmtree(tmp_dir)

        return result

    # ------------------------------------------------------------------
    # Internal: register masks across multiple reference frames
    # ------------------------------------------------------------------

    def _register_multi_frame_masks(
        self, video_pred, state, references: list,
    ) -> dict[int, str]:
        """
        Register organ masks from K reference images on frames 0..K-1.

        Each organ gets a unique obj_id (consistent across frames).
        If ref1 has {left_lung, heart} and ref2 has {left_lung, right_lung},
        then left_lung=obj_id 1, heart=2, right_lung=3. Masks are registered
        on whichever frames have them.

        Returns
        -------
        dict[int, str]
            obj_id -> organ_name mapping for result collection.
        """
        # Build a global organ_name -> obj_id mapping across all references
        all_organ_names: list[str] = []
        for ref in references:
            for organ_name in ref.masks:
                if organ_name not in all_organ_names:
                    all_organ_names.append(organ_name)

        organ_to_obj_id = {
            name: idx + 1 for idx, name in enumerate(all_organ_names)
        }
        obj_id_to_organ = {v: k for k, v in organ_to_obj_id.items()}

        # Register masks on each frame
        for frame_idx, ref in enumerate(references):
            for organ_name, mask in ref.masks.items():
                obj_id = organ_to_obj_id[organ_name]
                video_pred.add_new_mask(
                    inference_state=state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=mask.astype(np.float32),
                )

        print(f"    [VIDEO] Registered {len(all_organ_names)} organs "
              f"across {len(references)} reference frames: "
              f"{', '.join(all_organ_names)}")

        return obj_id_to_organ

    # ------------------------------------------------------------------
    # Internal: collect results from a single frame
    # ------------------------------------------------------------------

    def _collect_frame_results(
        self,
        video_res_masks: dict,
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
                print(f"    [VIDEO] Empty mask for '{organ}', skipping")
                continue

            confidence = self._logits_to_confidence(logits, binary_mask)

            objects.append(SegmentedObject(
                mask=binary_mask,
                source_image=source_image,
                confidence=confidence,
                label=organ_names_by_obj_id.get(obj_id, "unknown"),
            ))

        return objects

    # ------------------------------------------------------------------
    # Internal: predictor construction
    # ------------------------------------------------------------------

    def _get_video_predictor(self):
        if self._video_predictor is None:
            self._video_predictor = self._build_video_predictor()
        return self._video_predictor

    def _build_predictor(self) -> SAM2ImagePredictor:
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        ckpt = os.path.join(project_root, _CKPT_PATH)
        medsam2_root = os.path.join(project_root, "weights", "MedSAM2")
        original_cwd = os.getcwd()
        os.chdir(medsam2_root)
        try:
            model = build_sam2(_HYDRA_CFG, ckpt, device=self.device)
        finally:
            os.chdir(original_cwd)
        return SAM2ImagePredictor(model)

    def _build_video_predictor(self):
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        ckpt = os.path.join(project_root, _CKPT_PATH)
        medsam2_root = os.path.join(project_root, "weights", "MedSAM2")
        original_cwd = os.getcwd()
        os.chdir(medsam2_root)
        try:
            predictor = build_sam2_video_predictor(
                _HYDRA_CFG, ckpt, device=self.device
            )
        finally:
            os.chdir(original_cwd)
        print("[MedSAM2] Video predictor loaded.")
        return predictor

    def _predict_all_points(self, points: np.ndarray) -> list:
        label = np.array([1], dtype=np.int32)
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

    @staticmethod
    def _logits_to_confidence(
        logits: np.ndarray, binary_mask: np.ndarray
    ) -> float:
        fg_logits = logits[binary_mask]
        if len(fg_logits) == 0:
            return 0.0
        sigmoid_vals = 1.0 / (1.0 + np.exp(-np.clip(fg_logits, -20, 20)))
        return float(sigmoid_vals.mean())