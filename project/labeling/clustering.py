"""
Clustering-based labeling for segmented objects.

Supports KMeans, DBSCAN, and HDBSCAN. HDBSCAN supports adaptive parameters
computed as fractions of the dataset size for truly unsupervised operation.

Labels are numeric cluster IDs. Noise points (DBSCAN/HDBSCAN) receive
cluster_id -1 and confidence 0.0. No semantic organ name is assigned
at this stage.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

from project.core.interfaces import Labeler
from project.core.data_types import SegmentedObject, LabeledObject


FEATURE_NAMES = [
    # --- Kervadec et al. (original 6) ---
    "V", "Cx", "Cy", "Dx", "Dy", "L",
    # --- Shape descriptors ---
    "ecc", "solidity", "extent", "compact",
    # --- Hu moments (log-transformed) ---
    "hu0", "hu1", "hu2",
    # --- Intensity statistics ---
    "intensity_mean", "intensity_std",
    # --- Orientation ---
    "orientation",
]
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
    # Absolute values (used directly if set)
    min_cluster_size: int | None = None
    min_samples: int | None = None
    # Fraction-based values (computed from dataset size at runtime)
    min_cluster_size_fraction: float | None = None
    min_samples_fraction: float | None = None

    def resolve(self, num_images: int) -> "HDBSCANConfig":
        """
        Compute absolute values from fractions if not explicitly set.

        Fractions are relative to num_images. Absolute values take precedence
        over fractions when both are specified.

        Parameters
        ----------
        num_images : int
            Number of images in the dataset (not number of objects).

        Returns
        -------
        HDBSCANConfig
            New config with resolved absolute values and fractions set to None.
        """
        mcs = self.min_cluster_size
        ms = self.min_samples

        if mcs is None and self.min_cluster_size_fraction is not None:
            mcs = max(2, int(self.min_cluster_size_fraction * num_images))

        if ms is None and self.min_samples_fraction is not None:
            ms = max(1, int(self.min_samples_fraction * num_images))

        # Fallback defaults if nothing was specified
        if mcs is None:
            mcs = 2
        if ms is None:
            ms = 1

        return HDBSCANConfig(min_cluster_size=mcs, min_samples=ms)


@dataclass
class EmbeddingConfig:
    """Configuration for SAM2 encoder embeddings in clustering."""
    enabled: bool = False
    reduction: str | None = "pca"        # "pca", "umap", or None
    n_components: int = 16         # target dimensions after reduction (ignored if none)
    weight: float = 0.5            # relative weight vs moments (for weighted concat)
    # UMAP-specific (ignored if reduction != "umap")
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0

    def __post_init__(self):
        if self.reduction not in ("pca", "umap", None):
            raise ValueError(f"reduction must be 'pca', 'umap', or None, got '{self.reduction}'")


@dataclass
class ClusteringConfig:
    algorithm: ClusteringAlgorithm = ClusteringAlgorithm.KMEANS
    # None = no moment features (embeddings-only mode).
    # Empty list [] = use all FEATURE_NAMES (default).
    features: list[str] | None = field(default_factory=list)
    kmeans: KMeansConfig = field(default_factory=KMeansConfig)
    standardize: bool = False
    dbscan: DBSCANConfig = field(default_factory=DBSCANConfig)
    hdbscan: HDBSCANConfig = field(default_factory=HDBSCANConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    def __post_init__(self):
        if isinstance(self.algorithm, str):
            self.algorithm = ClusteringAlgorithm(self.algorithm)
        if isinstance(self.kmeans, dict):
            self.kmeans = KMeansConfig(**self.kmeans)
        if isinstance(self.dbscan, dict):
            self.dbscan = DBSCANConfig(**self.dbscan)
        if isinstance(self.hdbscan, dict):
            self.hdbscan = HDBSCANConfig(**self.hdbscan)
        if isinstance(self.embedding, dict):
            self.embedding = EmbeddingConfig(**self.embedding)

        # features=None means embeddings-only (no moment features).
        # features=[] (empty list) means use all available features (default).
        if self.features is None:
            # Explicit None: embeddings-only mode, no moment features.
            if not self.embedding.enabled:
                raise ValueError(
                    "features=None (embeddings-only mode) requires "
                    "embedding.enabled=True. Either set features to a "
                    "non-empty list or enable embeddings."
                )
            self.features = []
        elif isinstance(self.features, list) and len(self.features) == 0:
            # Empty list default: use all features.
            self.features = list(FEATURE_NAMES)

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
        self._scaler = StandardScaler() if config.standardize else None
        self._embedding_reducer = None  # Fitted during fit()

    @classmethod
    def from_config(cls, yaml_path: str) -> "ClusteringLabeler":
        """Load hyperparameters from a project YAML config file."""
        import yaml
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        config = ClusteringConfig(**raw.get("labeler", {}))
        return cls(config)

    # ------------------------------------------------------------------
    # Adaptive parameter resolution
    # ------------------------------------------------------------------

    def resolve_adaptive_params(self, num_images: int) -> None:
        """
        Resolve fraction-based HDBSCAN parameters using the dataset size.

        Must be called before fit() when using min_cluster_size_fraction
        or min_samples_fraction. Rebuilds the internal sklearn model with
        the resolved absolute values.

        No-op if the algorithm is not HDBSCAN or no fractions are configured.

        Parameters
        ----------
        num_images : int
            Number of images in the dataset.
        """
        if self.config.algorithm != ClusteringAlgorithm.HDBSCAN:
            return

        hcfg = self.config.hdbscan
        if hcfg.min_cluster_size_fraction is None and hcfg.min_samples_fraction is None:
            return

        resolved = hcfg.resolve(num_images)
        self.config.hdbscan = resolved
        self._model = self._build_model()

        print(f"  HDBSCAN adaptive resolution ({num_images} images): "
              f"min_cluster_size={resolved.min_cluster_size}, "
              f"min_samples={resolved.min_samples}")

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
        X = self._extract_feature_matrix(objects, fitting=True)
        print(f"Clustering matrix: {X.shape} "
              f"({len(self._feature_indices)} moments"
              f"{f' + {X.shape[1] - len(self._feature_indices)} embedding dims' if self.config.embedding.enabled else ''})")
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
            mcs = self.config.hdbscan.min_cluster_size or 2
            ms = self.config.hdbscan.min_samples or 1
            return HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
            )
        raise NotImplementedError(f"Algorithm '{self.config.algorithm}' is not implemented.")

    def _extract_feature_matrix(
        self, objects: list[SegmentedObject], fitting: bool = False,
    ) -> np.ndarray:
        """
        Build the feature matrix for clustering.

        Combines moment features (selected by config.features) and optionally
        SAM2 embeddings (reduced via PCA/UMAP). Supports three modes:
        - Moments only: config.features is non-empty, embedding.enabled=False
        - Embeddings only: config.features is empty, embedding.enabled=True
        - Hybrid: both non-empty features and embedding.enabled=True

        Parameters
        ----------
        objects : list[SegmentedObject]
            Objects with extracted features (and optionally embeddings).
        fitting : bool
            If True, fit scaler/reducer on this data. If False, transform only.

        Returns
        -------
        np.ndarray
            Feature matrix of shape (N, D) where D depends on the mode.
        """
        selected = None

        # --- Moment features (skip if no features selected) ---
        if self._feature_indices:
            missing = [i for i, obj in enumerate(objects) if obj.features is None]
            if missing:
                raise ValueError(
                    f"Objects at indices {missing} have no features. "
                    "Run a FeatureExtractor before clustering."
                )

            matrix = np.stack([obj.features for obj in objects])  # (N, 16)
            selected = matrix[:, self._feature_indices]           # (N, n_selected)

        # --- Embedding features ---
        if self.config.embedding.enabled:
            emb_cfg = self.config.embedding
            missing_emb = [i for i, obj in enumerate(objects) if obj.embedding is None]
            if missing_emb:
                raise ValueError(
                    f"Embedding enabled but objects at indices {missing_emb} "
                    "have no embedding. Extract embeddings before clustering."
                )

            embeddings = np.stack([obj.embedding for obj in objects])  # (N, 256)

            if not emb_cfg.reduction:
                # Use raw embeddings without dimensionality reduction
                reduced = embeddings
            elif fitting:
                self._embedding_reducer = self._build_reducer(emb_cfg, n_samples=len(objects))
                reduced = self._embedding_reducer.fit_transform(embeddings)
            else:
                if self._embedding_reducer is None:
                    raise RuntimeError("Embedding reducer not fitted.")
                reduced = self._embedding_reducer.transform(embeddings)

            # Concatenate or use standalone
            if selected is not None:
                selected = np.hstack([selected, reduced])
            else:
                selected = reduced

        if selected is None:
            raise ValueError(
                "No features available for clustering: both moment features "
                "and embeddings are disabled. Set config.features to a "
                "non-empty list or enable embeddings."
            )

        # --- Standardization (applied to the full concatenated vector) ---
        if self._scaler is not None:
            if fitting:
                return self._scaler.fit_transform(selected)
            return self._scaler.transform(selected)
        return selected

    def _build_reducer(self, emb_cfg: EmbeddingConfig, n_samples: int):
        """Build a dimensionality reduction model for embeddings."""
        if emb_cfg.reduction == "pca":
            from sklearn.decomposition import PCA
            n_components = min(emb_cfg.n_components, n_samples)
            if n_components < emb_cfg.n_components:
                print(
                    f"  [WARNING] PCA clipping n_components "
                    f"{emb_cfg.n_components} -> {n_components} "
                    f"(only {n_samples} samples available)"
                )
            return PCA(n_components=n_components, random_state=42)
        elif emb_cfg.reduction == "umap":
            import umap
            return umap.UMAP(
                n_components=emb_cfg.n_components,
                n_neighbors=emb_cfg.umap_n_neighbors,
                min_dist=emb_cfg.umap_min_dist,
                random_state=42,
            )

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