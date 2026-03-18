import yaml
import argparse
from pathlib import Path

from project.data_io.reader import MedicalImageReader
from project.data_io.utils import load_image_paths
from project.segmentation.medsam2 import MedSAM2Segmenter, MedSAM2Config
from project.feature_extraction.moments import MomentFeatureExtractor
from project.labeling.clustering import ClusteringLabeler, ClusteringConfig
from project.labeling.clustering_filter import ClusterFilter, ClusterFilterConfig
from project.core.pipeline import Pipeline
from project.evaluation.visualizer import save_visualization


def main(config_path: str):
    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    image_paths = load_image_paths(
        cfg["dataset"]["name"],
        cfg["dataset"]["extensions"],
    )

    max_images = cfg["dataset"].get("max_images")
    if max_images is not None:
        image_paths = image_paths[:max_images]

    if not image_paths:
        raise FileNotFoundError(f"No images found in {cfg['dataset']['images_dir']}")

    print(f"Found {len(image_paths)} images.")

    # ------------------------------------------------------------------
    # Build components
    # ------------------------------------------------------------------
    reader    = MedicalImageReader()
    segmenter = MedSAM2Segmenter(MedSAM2Config(**cfg["segmenter"]))
    extractor = MomentFeatureExtractor()
    labeler   = ClusteringLabeler(ClusteringConfig(**cfg["labeler"]))

    pipeline = Pipeline(reader, segmenter, extractor, labeler)

    # ------------------------------------------------------------------
    # Phase 1: segment + extract features for all images
    # ------------------------------------------------------------------
    print("Phase 1: segmenting and extracting features...")

    all_objects = []
    objects_by_image = {}

    for idx, path in enumerate(image_paths):
        print(f"Processing image {idx+1}/{len(image_paths)}: {path.name}")
        image = reader.load(str(path))
        objects = segmenter.segment(image)
        print(f"Image {path.name} segmented into {len(objects)} objects.")

        valid_objects = []
        for obj in objects:
            try:
                obj.features = extractor.extract(obj)
                valid_objects.append(obj)
            except ValueError as e:
                print(f"    [SKIP] obj {obj.id}: {e}")

        if valid_objects:
            objects_by_image[path] = valid_objects
            all_objects.extend(valid_objects)
        else:
            print(f"  [SKIP] {path.name}: no valid objects found")

    print(f"Total objects: {len(all_objects)}")

    # ------------------------------------------------------------------
    # Phase 2: fit labeler on all objects
    # ------------------------------------------------------------------
    print("Phase 2: fitting labeler...")
    labeler.fit(all_objects)

    # ------------------------------------------------------------------
    # Phase 3: label per image
    # ------------------------------------------------------------------
    print("Phase 3: labeling...")

    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    labeled_by_image = {}
    for path, objects in objects_by_image.items():
        labeled = labeler.label(objects)
        labeled_by_image[path] = labeled

        for obj in labeled:
            print(
                f"  {path.name} | "
                f"obj {obj.segmented_object.id} → "
                f"{obj.organ_name} (confidence: {obj.labeling_confidence:.2f})"
            )

    print("Total objects:", len(all_objects))
    print("Unique ids:", len(set(obj.id for obj in all_objects)))
    
    # ------------------------------------------------------------------
    # Phase 4: filter noise clusters
    # ------------------------------------------------------------------
    print("Phase 4: filtering noise clusters...")
    cluster_filter = ClusterFilter(ClusterFilterConfig(**cfg["cluster_filter"]))
    labeled_by_image = cluster_filter.filter(labeled_by_image)

    # Visualization
    for path, labeled in labeled_by_image.items():
        save_visualization(path, labeled, results_dir)

    import numpy as np
    X = np.stack([obj.features for obj in all_objects])
    print("Feature matrix shape:", X.shape)
    print("Min:", X.min(axis=0))
    print("Max:", X.max(axis=0))
    print("Std:", X.std(axis=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()
    main(args.config)