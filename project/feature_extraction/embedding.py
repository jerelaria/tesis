"""
SAM2 encoder embedding extraction via masked average pooling.

Extracts a per-object feature vector from SAM2's image encoder output.
The image encoder produces a spatial feature map for the entire image;
this module crops it to each object's mask and averages, yielding a
dense semantic descriptor per object.

Architecture (what SAM2 gives us):
    Image (1024x1024) -> Hiera encoder -> feature map (1, 256, 64, 64)

    Each "pixel" in the 64x64 feature map corresponds to a ~16x16
    patch of the original image and carries 256 channels of learned
    semantic information.

Masked average pooling (what we do):
    1. Resize object mask from (H, W) -> (64, 64) via nearest-neighbor
    2. Multiply feature map elementwise by resized mask
    3. Average over spatial dims -> vector of shape (256,)

This vector captures what SAM2's encoder "thinks" about the region
covered by the mask: texture, structure, context — information that
classical moments cannot capture.

The raw 256-d embedding is stored in SegmentedObject.embedding.
Dimensionality reduction (PCA) and concatenation with moments happen
in the clustering labeler, not here.
"""

import numpy as np
import torch
import torch.nn.functional as F

from project.core.data_types import SegmentedObject


def extract_sam2_embedding(
    obj: SegmentedObject,
    image_embed: torch.Tensor,
) -> np.ndarray:
    """
    Extract a per-object embedding via masked average pooling.

    Parameters
    ----------
    obj : SegmentedObject
        Must have `mask` (H, W) bool.
    image_embed : torch.Tensor
        SAM2 image encoder output, shape (1, C, H_feat, W_feat).
        Typically (1, 256, 64, 64).

    Returns
    -------
    np.ndarray (C,) float32
        Pooled embedding vector for this object.

    Raises
    ------
    ValueError
        If mask is empty (no pixels to pool over).
    """
    mask = obj.mask.astype(np.float32)

    if not mask.any():
        raise ValueError("Empty mask, cannot extract embedding")

    feat_h, feat_w = image_embed.shape[-2:]

    # Resize mask to feature map resolution
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    mask_resized = F.interpolate(
        mask_tensor,
        size=(feat_h, feat_w),
        mode="nearest",
    ).squeeze(0).squeeze(0)  # (H_feat, W_feat)

    # Move to same device as features
    mask_resized = mask_resized.to(image_embed.device)

    # Masked average pooling: (C, H, W) * (H, W) -> mean over spatial -> (C,)
    feat_map = image_embed.squeeze(0)  # (C, H_feat, W_feat)
    masked_feats = feat_map * mask_resized.unsqueeze(0)  # broadcast over C
    pooled = masked_feats.sum(dim=(1, 2)) / (mask_resized.sum() + 1e-8)  # (C,)

    return pooled.detach().cpu().numpy().astype(np.float32)