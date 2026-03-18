import os
import sys
import json
import cv2
import glob
import numpy as np
import torch
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path("./ChestXMask").resolve()
INPUT_DIR = PROJECT_ROOT / "data" / "images"
OUTPUT_DIR = PROJECT_ROOT / "data" / "masks"

os.chdir("MedSAM2")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Configuration
INPUT_DIR = "../data/xray/"
OUTPUT_DIR = "../output_masks/"
CHECKPOINT = "checkpoints/MedSAM2_latest.pt"
CONFIG = "configs/sam2.1_hiera_t512.yaml"

# SAM Inference Hyperparameters
GRID_PTS     = 10
SCORE_MIN    = 0.78
AREA_FRAC    = 0.001
IOU_NMS      = 0.55

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Utils (should be in a separate utils.py in the future)
def iou_mask(a, b):
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return inter / (uni + 1e-8)

def nms_by_iou(masks, scores, iou_thr=0.5): # non maximum suppression for masks based on IoU
    if not masks: return []
    order = np.argsort(scores)[::-1]
    masks = [masks[i] for i in order]
    scores = [scores[i] for i in order]
    kept = []
    suppressed = np.zeros(len(masks), dtype=bool)
    for i in range(len(masks)):
        if suppressed[i]: continue
        kept.append((masks[i], scores[i]))
        for j in range(i+1, len(masks)):
            if not suppressed[j] and iou_mask(masks[i], masks[j]) >= iou_thr:
                suppressed[j] = True
    return kept

def grid_points(W, H, grid_pts):
    xs = np.linspace(W*0.02, W*0.98, grid_pts).astype(int)
    ys = np.linspace(H*0.02, H*0.98, grid_pts).astype(int)
    return [(x,y) for y in ys for x in xs]

# Model initialization
device = "cuda" if torch.cuda.is_available() else "cpu"
sam2_model = build_sam2(CONFIG, CHECKPOINT, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# Fetch all image paths
image_paths = glob.glob(os.path.join(INPUT_DIR, "*.png"))

print(f"Found {len(image_paths)} images. Starting extraction...")

for img_path in image_paths[:1]:
    filename = os.path.basename(img_path).split('.')[0]
    image_pil = Image.open(img_path).convert("RGB")
    image_np = np.array(image_pil)
    H, W = image_np.shape[:2]
    AREA_MIN = max(50, int(AREA_FRAC * H * W))
    
    predictor.set_image(image_np)
    
    cand_masks, cand_scores = [], []
    pts = grid_points(W, H, GRID_PTS)
    
    for (x, y) in pts:
        msks, scs, _ = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True
        )

        for m, s in zip(msks, scs):
            if float(s) >= SCORE_MIN and int(m.sum()) >= AREA_MIN:
                cand_masks.append(m.astype(bool))
                cand_scores.append(float(s))
    
    # Apply NMS to remove overlapping duplicates
    kept = nms_by_iou(cand_masks, cand_scores, iou_thr=IOU_NMS)
    
    if kept:
            # Formato:
            # {
            #   "0": [[x1, y1], [x2, y2], ...],  # Contorno del órgano 0
            #   "1": [[x1, y1], [x2, y2], ...],  # Contorno del órgano 1
            #   ...
            # }
            landmarks_data = {}

            for i, (mask, score) in enumerate(kept):
                class_id = str(i)
                
                # Convert boolean mask to uint8 for contour detection
                mask_uint8 = (mask * 255).astype(np.uint8)
                
                # RETR_EXTERNAL: Only retrieve external contours
                # CHAIN_APPROX_SIMPLE: Compresses horizontal, vertical, and diagonal segments
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Take the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Filter out small contours
                    if cv2.contourArea(largest_contour) > 100: # This should be another hyperparameter
                        # Convert contour to list of points
                        points_list = largest_contour.squeeze().tolist()
                        
                        # Ensure points_list is a list of points
                        if isinstance(points_list, list) and len(points_list) > 2:
                            landmarks_data[class_id] = points_list
            
            # Save landmarks data to JSON
            json_filename = f"{filename}.json"
            save_path = os.path.join(OUTPUT_DIR, json_filename)
            
            with open(save_path, 'w') as f:
                json.dump(landmarks_data, f)
                
            print(f"JSON saved for {filename} with {len(landmarks_data)} organs.")
            
    else:
        print(f"No masks found for {filename}")

print("Done!")