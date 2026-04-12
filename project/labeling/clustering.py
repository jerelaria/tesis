import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

from project.core.interfaces import Labeler
from project.core.data_types import SegmentedObject, LabeledObject


# Feature names must match the order produced by MomentFeatureExtractor
FEATURE_NAMES = ["V", "Cx", "Cy", "Dx", "Dy", "L"]
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


class ClusteringAlgorithm(Enum):
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HDBSCAN = "hdbscan"

@dataclass
class KMeansConfig:
    n_clusters: int = 2
    random_state: int = 42

@dataclass
class DBSCANConfig:
    eps: float = 0.5
    min_samples: int = 2

@dataclass
class HDBSCANConfig:
    min_cluster_size: int = 2
    min_samples: int = 1

@dataclass
class ClusteringConfig:
    algorithm: ClusteringAlgorithm = ClusteringAlgorithm.KMEANS
    features: list[str] = field(default_factory=list)
    kmeans: KMeansConfig = field(default_factory=KMeansConfig)
    dbscan: DBSCANConfig = field(default_factory=DBSCANConfig)
    hdbscan: HDBSCANConfig = field(default_factory=HDBSCANConfig)

    def __post_init__(self):
        if isinstance(self.algorithm, str):
            self.algorithm = ClusteringAlgorithm(self.algorithm)
        if isinstance(self.kmeans, dict):
            self.kmeans = KMeansConfig(**self.kmeans)
        if isinstance(self.dbscan, dict):
            self.dbscan = DBSCANConfig(**self.dbscan)
        if isinstance(self.hdbscan, dict):
            self.hdbscan = HDBSCANConfig(**self.hdbscan)
        if not self.features:
            self.features = FEATURE_NAMES
        unknown = [f for f in self.features if f not in FEATURE_INDEX]
        if unknown:
            raise ValueError(f"Unknown feature names: {unknown}. Valid: {FEATURE_NAMES}")


class ClusteringLabeler(Labeler):
    """
    Assigns organ labels to segmented objects using clustering over shape features.

    Supports KMeans, DBSCAN and HDBSCAN. Labels are numeric cluster IDs.
    Noise points (DBSCAN/HDBSCAN) receive cluster_id -1 and confidence 0.0.
    No semantic organ name is assigned at this stage.

    IMPORTANT: fit() must receive objects from ALL images in the dataset,
    as label() performs a lookup over the mapping built during fit().
    Objects not seen during fit() cannot be labeled.

    Parameters
    ----------
    config : ClusteringConfig
        Clustering hyperparameters and feature selection.

    Example
    -------
    >>> labeler = ClusteringLabeler(ClusteringConfig(features=["Cx", "Cy"]))
    >>> labeler.fit(all_objects_from_all_images)
    >>> labeled = labeler.label(objects_from_one_image)
    """

    def __init__(self, config: ClusteringConfig = ClusteringConfig()):
        self.config = config
        self._model = self._build_model()
        self._is_fitted = False
        self._feature_indices = [FEATURE_INDEX[f] for f in config.features]
        self._scaler = StandardScaler()

    @classmethod
    def from_config(cls, yaml_path: str) -> "ClusteringLabeler":
        """Load hyperparameters from a project YAML config file."""
        import yaml
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        config = ClusteringConfig(**raw.get("labeler", {}))
        return cls(config)

    # ------------------------------------------------------------------
    # Labeler interface
    # ------------------------------------------------------------------

    def fit(self, objects: list[SegmentedObject]) -> None:
        """
        Fit the clustering model on feature vectors from all provided objects.
        Should be called once with objects aggregated from the full dataset.

        Raises
        ------
        ValueError
            If any object has no features extracted.
        """
        X = self._extract_feature_matrix(objects, fitting=False)
        cluster_ids = self._model.fit_predict(X)
        print("Unique clusters:", np.unique(cluster_ids))
        print("Noise points:", (cluster_ids == -1).sum())
        print("Total points:", len(cluster_ids))
        self._id_to_cluster = {obj.id: int(cid) for obj, cid in zip(objects, cluster_ids)}
        self._is_fitted = True

    def label(self, objects: list[SegmentedObject]) -> list[LabeledObject]:
        """
        Assign a cluster ID to each object using the mapping built during fit().
        All objects must have been present in the fit() call.

        Raises
        ------
        RuntimeError
            If called before fit().
        ValueError
            If any object was not seen during fit().
        """
        if not self._is_fitted:
            raise RuntimeError("ClusteringLabeler must be fitted before calling label().")

        unseen = [obj.id for obj in objects if obj.id not in self._id_to_cluster]
        if unseen:
            raise ValueError(
                f"Objects with ids {unseen} were not seen during fit(). "
                "Call fit() with all objects before calling label()."
            )

        X = self._extract_feature_matrix(objects)
        cluster_ids = np.array([self._id_to_cluster[obj.id] for obj in objects])
        confidences = self._compute_confidences(X, cluster_ids)

        labeled = []
        for obj, cluster_id, confidence in zip(objects, cluster_ids, confidences):
            labeled.append(LabeledObject(
                segmented_object=obj,
                organ_id=int(cluster_id),
                organ_name=f"cluster_{cluster_id}",
                labeling_confidence=float(confidence),
                method_used=f"clustering_{self.config.algorithm.value}",
            ))

        return labeled

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self):
        """Factory: returns the sklearn clustering model for the configured algorithm."""
        if self.config.algorithm == ClusteringAlgorithm.KMEANS:
            return KMeans(
                n_clusters=self.config.kmeans.n_clusters,
                random_state=self.config.kmeans.random_state,
                n_init="auto",
            )
        if self.config.algorithm == ClusteringAlgorithm.DBSCAN:
            return DBSCAN(
                eps=self.config.dbscan.eps,
                min_samples=self.config.dbscan.min_samples,
            )
        if self.config.algorithm == ClusteringAlgorithm.HDBSCAN:
            return HDBSCAN(
                min_cluster_size=self.config.hdbscan.min_cluster_size,
                min_samples=self.config.hdbscan.min_samples,
            )
        raise NotImplementedError(f"Algorithm '{self.config.algorithm}' is not implemented.")

    def _extract_feature_matrix(self, objects: list[SegmentedObject], fitting: bool = False) -> np.ndarray:
        missing = [i for i, obj in enumerate(objects) if obj.features is None]
        if missing:
            raise ValueError(
                f"Objects at indices {missing} have no features. "
                "Run a FeatureExtractor before clustering."
            )

        matrix = np.stack([obj.features for obj in objects])  # (N, 6)
        selected = matrix[:, self._feature_indices]           # (N, n_selected)

        if fitting:
            return self._scaler.fit_transform(selected)
        return matrix[:, self._feature_indices]  

    def _compute_confidences(self, X: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
        """Dispatcher: routes to the appropriate confidence computation method."""
        if self.config.algorithm == ClusteringAlgorithm.KMEANS:
            return self._confidences_kmeans(X, cluster_ids)
        return self._confidences_density_based(X, cluster_ids)
    
    def _confidences_kmeans(self, X: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
        """
        Confidence score in [0, 1] based on distance to assigned centroid
        relative to all centroids.

        Confidence = 1 - (dist_to_assigned / sum_of_all_distances)
        """
        centroids = self._model.cluster_centers_
        confidences = np.zeros(len(X))

        for i, (x, cid) in enumerate(zip(X, cluster_ids)):
            distances = np.linalg.norm(centroids - x, axis=1)
            total = distances.sum()
            if total < 1e-8:
                confidences[i] = 1.0
            else:
                confidences[i] = 1.0 - (distances[cid] / total)

        return confidences

    def _confidences_density_based(self, X: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
        """
        Confidence score in [0, 1] for density-based algorithms (DBSCAN, HDBSCAN).
        Noise points (cluster_id == -1) receive confidence 0.0.
        Non-noise points receive confidence 1.0 as density-based algorithms
        have no notion of soft assignment.
        """
        return np.where(cluster_ids == -1, 0.0, 1.0)