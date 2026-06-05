import logging
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

from project.core.interfaces import Segmenter
from project.core.data_types import MedicalImage, SegmentedObject
from project.segmentation.utils import to_uint8, nms, make_point_grid
from project.segmentation.medsam2.model_loading import (
    build_image_predictor,
    build_video_predictor,
)
from project.segmentation.medsam2.video import (
    _video_session,
    register_multi_frame_masks,
    collect_frame_results,
    logits_to_confidence,
)


@dataclass
class MedSAM2Config:
    device: str = "cuda"
    grid_side: int = 6
    score_threshold: float = 0.50
    iou_threshold: float = 0.50


class MedSAM2Segmenter(Segmenter):
    """
    Segment a MedicalImage using MedSAM2.

    Three public methods:

    1. segment() — grid of points, image predictor. Unsupervised Phase 1 only.

    2. segment_with_video_prompts() — (K+1)-frame video: K reference frames
       (each with multiple organ masks) + 1 target. Independent propagation.

    3. segment_batch_iterative() — (K+N)-frame video: K references + N targets.
       Memory accumulates from references AND previously segmented targets.

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
        self._predictor = build_image_predictor(self.device)
        self._video_predictor = None

    @classmethod
    def from_config(cls, yaml_path: str) -> "MedSAM2Segmenter":
        import yaml
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        config = MedSAM2Config(**raw.get("segmenter", {}))
        return cls(config)

    def encode_image(self, image: MedicalImage) -> torch.Tensor:
        """Run image through Hiera encoder; returns embedding (1, 256, 64, 64)."""
        img_uint8 = to_uint8(image.volume)
        self._predictor.set_image(img_uint8)
        return self._predictor._features["image_embed"]

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

    def segment_with_video_prompts(
        self,
        target_image: MedicalImage,
        references: list,
    ) -> list[SegmentedObject]:
        if not references:
            return []

        video_pred = self._get_video_predictor()
        K = len(references)
        ref_uint8s = [to_uint8(r.volume) for r in references]
        tgt_uint8 = to_uint8(target_image.volume)

        def _register(vp, st):
            return register_multi_frame_masks(vp, st, references)

        objects = []
        with _video_session(video_pred, ref_uint8s, [tgt_uint8], _register) as (state, organ_map):
            for frame_idx, obj_ids, video_res_masks in video_pred.propagate_in_video(state):
                if frame_idx == K:
                    objects.extend(
                        collect_frame_results(video_res_masks, obj_ids, organ_map, target_image)
                    )
        return objects

    def segment_with_multi_reference(
        self,
        target_image: MedicalImage,
        reference_entries: list[tuple[np.ndarray, np.ndarray]],
        organ_name: str,
    ) -> SegmentedObject | None:
        """Single-object (K+1)-frame video session for one organ.

        reference_entries: list of (volume_array, mask_array) pairs (K frames).
        Returns a SegmentedObject for the target frame, or None if the mask is empty.
        """
        if not reference_entries:
            return None

        video_pred = self._get_video_predictor()
        obj_id = 1
        K = len(reference_entries)
        ref_uint8s = [to_uint8(vol) for vol, _ in reference_entries]
        tgt_uint8 = to_uint8(target_image.volume)

        def _register(vp, st):
            for fi, (_, mask) in enumerate(reference_entries):
                vp.add_new_mask(
                    inference_state=st,
                    frame_idx=fi,
                    obj_id=obj_id,
                    mask=mask.astype(np.float32),
                )
            return {obj_id: organ_name}

        result = None
        with _video_session(video_pred, ref_uint8s, [tgt_uint8], _register) as (state, organ_map):
            for frame_idx, obj_ids, video_res_masks in video_pred.propagate_in_video(state):
                if frame_idx == K:
                    objects = collect_frame_results(
                        video_res_masks, obj_ids, organ_map, target_image
                    )
                    if objects:
                        result = objects[0]
        return result

    def segment_batch_iterative_per_cluster(
        self,
        target_entries: list[tuple[Path, MedicalImage]],
        reference_entries: list[tuple[np.ndarray, np.ndarray]],
        organ_name: str,
    ) -> dict[Path, "SegmentedObject | None"]:
        """Iterative (K+N)-frame session for a single organ.

        Like segment_batch_iterative but for one organ only.
        reference_entries: list of (volume_array, mask_array) pairs (K frames).
        Memory accumulates from each target prediction into the next.
        Returns {path: SegmentedObject | None} for each target.
        """
        if not target_entries or not reference_entries:
            return {path: None for path, _ in target_entries}

        video_pred = self._get_video_predictor()
        obj_id = 1
        K = len(reference_entries)

        frame_to_path = {K + i: path for i, (path, _) in enumerate(target_entries)}
        frame_to_image = {K + i: img for i, (_, img) in enumerate(target_entries)}

        ref_uint8s = [to_uint8(vol) for vol, _ in reference_entries]
        tgt_uint8s = [to_uint8(img.volume) for _, img in target_entries]

        def _register(vp, st):
            for fi, (_, mask) in enumerate(reference_entries):
                vp.add_new_mask(
                    inference_state=st,
                    frame_idx=fi,
                    obj_id=obj_id,
                    mask=mask.astype(np.float32),
                )
            return {obj_id: organ_name}

        results: dict[Path, "SegmentedObject | None"] = {
            path: None for path, _ in target_entries
        }
        with _video_session(video_pred, ref_uint8s, tgt_uint8s, _register) as (state, organ_map):
            for frame_idx, obj_ids, video_res_masks in video_pred.propagate_in_video(state):
                if frame_idx not in frame_to_path:
                    continue
                path = frame_to_path[frame_idx]
                source_image = frame_to_image[frame_idx]
                objects = collect_frame_results(
                    video_res_masks, obj_ids, organ_map, source_image
                )
                if objects:
                    results[path] = objects[0]
        return results

    def segment_batch_iterative(
        self,
        target_entries: list[tuple[Path, MedicalImage]],
        references: list,
    ) -> dict[Path, list[SegmentedObject]]:
        if not target_entries or not references:
            return {}

        video_pred = self._get_video_predictor()
        K = len(references)

        frame_to_path = {K + i: path for i, (path, _) in enumerate(target_entries)}
        frame_to_image = {K + i: img for i, (_, img) in enumerate(target_entries)}

        ref_uint8s = [to_uint8(r.volume) for r in references]
        tgt_uint8s = [to_uint8(img.volume) for _, img in target_entries]

        def _register(vp, st):
            return register_multi_frame_masks(vp, st, references)

        total_frames = K + len(target_entries)
        logger.info(
            f"[VIDEO BATCH] {total_frames}-frame video "
            f"({K} refs + {len(target_entries)} targets)"
        )

        results: dict[Path, list[SegmentedObject]] = {
            path: [] for path, _ in target_entries
        }
        propagated_count = 0

        with _video_session(video_pred, ref_uint8s, tgt_uint8s, _register) as (state, organ_map):
            for frame_idx, obj_ids, video_res_masks in video_pred.propagate_in_video(state):
                if frame_idx not in frame_to_path:
                    continue
                path = frame_to_path[frame_idx]
                source_image = frame_to_image[frame_idx]
                frame_objects = collect_frame_results(
                    video_res_masks, obj_ids, organ_map, source_image
                )
                results[path].extend(frame_objects)
                propagated_count += len(frame_objects)

                target_idx = frame_idx - K
                n_targets = len(target_entries)
                if target_idx % 50 == 0 or target_idx == n_targets - 1:
                    logger.debug(f"[VIDEO BATCH] Target {target_idx+1}/{n_targets}")

        logger.info(
            f"[VIDEO BATCH] {propagated_count} objects across {len(target_entries)} images"
        )
        return results

    def _get_video_predictor(self):
        if self._video_predictor is None:
            self._video_predictor = build_video_predictor(self.device)
        return self._video_predictor

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
