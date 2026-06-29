"""
Tests for PrototypePropagator and build_unsupervised_references.

GPU is NOT used.  A StubVideoSegmenter replaces MedSAM2Segmenter; it captures
every call to segment_with_multi_reference and returns synthetic SegmentedObjects.

Conventions verified:
  - segment_with_multi_reference is called n_good_clusters × n_targets times.
  - Each call carries the correct organ_name (f"cluster_{cid}") and correct
    number of reference entries (min(k, n_prototypes_available)).
  - organ_id = sorted_position(label) + 1  (SAM2 rejects obj_id=0).
  - Output has one LabeledObject per cluster per target; method_used="propagation".
  - Clusters failing quality thresholds are absent from the result.
  - Memory composition exposes the required fields.
  - Third return element (debug_predictions) is [] when the flag is False.
"""
import numpy as np
import pytest
from pathlib import Path

from project.core.data_types import MedicalImage, SegmentedObject, LabeledObject
from project.data_io.few_shot_reader import FewShotReference
from project.pipeline.propagator import PropagationConfig, PrototypePropagator
from project.pipeline.reference_builder import (
    build_fewshot_references,
    build_unsupervised_references,
)
from project.segmentation.quality import ClusterQualityConfig, MaskSelectionConfig

H, W = 20, 20


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class StubVideoSegmenter:
    """Captures calls to segment_with_multi_reference and
    segment_batch_iterative_per_cluster; returns synthetic SegmentedObjects."""

    def __init__(self):
        self.multi_ref_call_count = 0
        self.multi_ref_calls: list[tuple[MedicalImage, str, int]] = []
        self.batch_per_cluster_call_count = 0
        self.batch_per_cluster_calls: list[tuple[str, int, int]] = []
        # kept so tests can verify it is never called
        self.call_count = 0
        self.batch_iterative_call_count = 0

    def segment_with_multi_reference(
        self,
        target_image: MedicalImage,
        reference_entries: list,
        organ_name: str,
    ) -> SegmentedObject:
        self.multi_ref_call_count += 1
        self.multi_ref_calls.append((target_image, organ_name, len(reference_entries)))
        return SegmentedObject(
            mask=np.ones((H, W), dtype=bool),
            source_image=target_image,
            confidence=0.8,
            label=organ_name,
        )

    def segment_batch_iterative_per_cluster(
        self,
        target_entries: list,
        reference_entries: list,
        organ_name: str,
    ) -> dict:
        self.batch_per_cluster_call_count += 1
        self.batch_per_cluster_calls.append(
            (organ_name, len(reference_entries), len(target_entries))
        )
        return {
            path: SegmentedObject(
                mask=np.ones((H, W), dtype=bool),
                source_image=source_image,
                confidence=0.8,
                label=organ_name,
            )
            for path, source_image in target_entries
        }

    def segment_with_video_prompts(
        self,
        target_image: MedicalImage,
        references: list,
    ) -> list[SegmentedObject]:
        self.call_count += 1
        return []

    def segment_batch_iterative(
        self,
        target_entries: list,
        references: list,
    ) -> dict:
        self.batch_iterative_call_count += 1
        return {}


class StubReader:
    def load(self, path: str) -> MedicalImage:
        return MedicalImage(
            volume=np.zeros((H, W, 3), dtype=np.float32),
            modality="synthetic",
            source_path=path,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_labeled(
    organ_id: int,
    labeling_confidence: float = 0.8,
    sam_confidence: float = 0.9,
    source_path: str = "/tmp/img.png",
) -> LabeledObject:
    image = MedicalImage(
        volume=np.zeros((H, W, 3), dtype=np.float32),
        modality="synthetic",
        source_path=source_path,
    )
    seg = SegmentedObject(
        mask=np.ones((H, W), dtype=bool),
        source_image=image,
        confidence=sam_confidence,
    )
    return LabeledObject(
        segmented_object=seg,
        organ_id=organ_id,
        organ_name=f"cluster_{organ_id}",
        labeling_confidence=labeling_confidence,
        method_used="test",
    )


def _build_dataset(
    n_images: int,
    cluster_ids: list[int],
    labeling_confidence: float = 0.8,
    sam_confidence: float = 0.9,
) -> dict[Path, list[LabeledObject]]:
    dataset: dict[Path, list[LabeledObject]] = {}
    for i in range(n_images):
        path = Path(f"/tmp/img_{i}.png")
        dataset[path] = [
            _make_labeled(
                organ_id=cid,
                labeling_confidence=labeling_confidence,
                sam_confidence=sam_confidence,
                source_path=str(path),
            )
            for cid in cluster_ids
        ]
    return dataset


def _permissive_config(k: int = 2) -> PropagationConfig:
    """Config that accepts all clusters (zero quality thresholds)."""
    return PropagationConfig(
        references_per_cluster=k,
        quality=ClusterQualityConfig(
            min_image_frequency=0.0,
            min_avg_labeling_confidence=0.0,
            min_avg_sam_confidence=0.0,
        ),
        mask_selection=MaskSelectionConfig(
            sam_score_weight=0.5,
            min_combined_score=0.0,
        ),
    )


def _build_and_propagate(
    labeled_by_image: dict[Path, list[LabeledObject]],
    config: PropagationConfig,
    target_paths: list[Path],
    stub: StubVideoSegmenter,
    reader: StubReader | None = None,
):
    """Call build_unsupervised_references then propagate. Returns 3 elements."""
    reader = reader or StubReader()
    propagator = PrototypePropagator(segmenter=stub, config=config)
    refs, scores = build_unsupervised_references(labeled_by_image, config, reader)
    return propagator.propagate(refs, scores, target_paths, reader)


# ---------------------------------------------------------------------------
# Per-cluster session discipline
# ---------------------------------------------------------------------------

class TestPerClusterSessions:

    def _run(self, cluster_ids, k, n_images=10, n_targets=3):
        labeled_by_image = _build_dataset(n_images, cluster_ids)
        stub = StubVideoSegmenter()
        config = _permissive_config(k=k)
        target_paths = [Path(f"/tmp/tgt_{i}.png") for i in range(n_targets)]
        _build_and_propagate(labeled_by_image, config, target_paths, stub)
        return stub

    def test_call_count_equals_clusters_times_targets(self):
        """segment_with_multi_reference called n_clusters × n_targets times."""
        n_clusters, n_targets = 3, 4
        stub = self._run(cluster_ids=[0, 1, 2], k=2, n_targets=n_targets)
        assert stub.multi_ref_call_count == n_clusters * n_targets

    def test_organ_names_are_correct(self):
        """Each call must carry the cluster's own organ_name."""
        stub = self._run(cluster_ids=[0, 1, 2], k=2, n_targets=2)
        called_organs = {organ for _, organ, _ in stub.multi_ref_calls}
        assert called_organs == {"cluster_0", "cluster_1", "cluster_2"}

    def test_reference_entries_count_respects_k(self):
        """len(reference_entries) == min(k, n_prototypes_available) per cluster."""
        k = 3
        stub = self._run(cluster_ids=[0, 1], k=k, n_images=10, n_targets=1)
        for _, organ_name, n_entries in stub.multi_ref_calls:
            assert n_entries == k, (
                f"{organ_name}: expected {k} entries, got {n_entries}"
            )

    def test_fewer_entries_when_prototypes_unavailable(self):
        """A cluster with only 1 prototype gets 1 entry even when k=3."""
        labeled_by_image = _build_dataset(n_images=10, cluster_ids=[0, 2])
        path_0 = Path("/tmp/img_0.png")
        labeled_by_image[path_0].append(
            _make_labeled(organ_id=1, source_path=str(path_0))
        )
        stub = StubVideoSegmenter()
        config = _permissive_config(k=3)
        _build_and_propagate(
            labeled_by_image, config, [Path("/tmp/tgt_0.png")], stub
        )
        # cluster_1 had only 1 prototype
        entries_by_organ = {
            organ: n_entries for _, organ, n_entries in stub.multi_ref_calls
        }
        assert entries_by_organ["cluster_1"] == 1
        assert entries_by_organ["cluster_0"] == 3
        assert entries_by_organ["cluster_2"] == 3


# ---------------------------------------------------------------------------
# Batch call discipline
# ---------------------------------------------------------------------------

def test_segment_called_n_clusters_times_per_target():
    """segment_with_multi_reference is called n_clusters × n_targets times total."""
    n_targets = 20
    n_clusters = 3
    labeled_by_image = _build_dataset(n_images=10, cluster_ids=[0, 1, 2])
    stub = StubVideoSegmenter()
    config = _permissive_config(k=3)
    target_paths = [Path(f"/tmp/tgt_{i}.png") for i in range(n_targets)]
    _build_and_propagate(labeled_by_image, config, target_paths, stub)
    assert stub.multi_ref_call_count == n_clusters * n_targets


def test_all_target_paths_sent_to_segmenter():
    """Every target_path must reach segment_with_multi_reference."""
    labeled_by_image = _build_dataset(n_images=10, cluster_ids=[0])
    stub = StubVideoSegmenter()
    config = _permissive_config(k=1)
    target_paths = [Path(f"/tmp/tgt_{i}.png") for i in range(7)]
    _build_and_propagate(labeled_by_image, config, target_paths, stub)
    sent_paths = {Path(img.source_path) for img, _, _ in stub.multi_ref_calls}
    assert sent_paths == set(target_paths)


# ---------------------------------------------------------------------------
# obj_id mapping
# ---------------------------------------------------------------------------

def test_obj_id_sorted_position_plus_one():
    """
    obj_id = sorted_position(label) + 1.
    For clusters {0,1,2}: labels sorted as cluster_0 < cluster_1 < cluster_2,
    so obj_ids are 1, 2, 3.
    For clusters {2,5,7}: similarly obj_ids 1, 2, 3.
    """
    for cluster_ids in [[0, 1, 2], [2, 5, 7]]:
        labeled_by_image = _build_dataset(n_images=10, cluster_ids=cluster_ids)
        stub = StubVideoSegmenter()
        config = _permissive_config(k=1)
        _, memory, _ = _build_and_propagate(
            labeled_by_image, config, [Path("/tmp/t.png")], stub
        )
        sorted_labels = sorted(f"cluster_{cid}" for cid in cluster_ids)
        expected = {label: i + 1 for i, label in enumerate(sorted_labels)}
        for entry in memory:
            assert entry["obj_id"] == expected[entry["label"]], (
                f"{entry['label']}: expected obj_id "
                f"{expected[entry['label']]}, got {entry['obj_id']}"
            )


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:

    def _propagate(self, cluster_ids, n_targets=5, k=2):
        labeled_by_image = _build_dataset(n_images=10, cluster_ids=cluster_ids)
        stub = StubVideoSegmenter()
        config = _permissive_config(k=k)
        target_paths = [Path(f"/tmp/tgt_{i}.png") for i in range(n_targets)]
        result, memory, _ = _build_and_propagate(
            labeled_by_image, config, target_paths, stub
        )
        return result, memory, stub

    def test_all_targets_present_as_keys(self):
        result, _, _ = self._propagate([0, 1], n_targets=5)
        assert len(result) == 5

    def test_one_labeled_object_per_cluster_per_target(self):
        result, _, _ = self._propagate([0, 1, 2], n_targets=4)
        expected_names = {"cluster_0", "cluster_1", "cluster_2"}
        for path, labeled in result.items():
            names_found = {obj.organ_name for obj in labeled}
            assert names_found == expected_names, (
                f"{path.name}: expected {expected_names}, got {names_found}"
            )

    def test_organ_id_is_sorted_label_position_plus_one(self):
        """organ_id = sorted_position(organ_name) + 1."""
        result, _, _ = self._propagate([0, 1], n_targets=3)
        sorted_names = sorted({"cluster_0", "cluster_1"})
        expected = {name: i + 1 for i, name in enumerate(sorted_names)}
        for labeled in result.values():
            for obj in labeled:
                assert obj.organ_id == expected[obj.organ_name], (
                    f"{obj.organ_name}: expected organ_id "
                    f"{expected[obj.organ_name]}, got {obj.organ_id}"
                )

    def test_method_used_is_propagation(self):
        result, _, _ = self._propagate([0, 1], n_targets=3)
        for labeled in result.values():
            for obj in labeled:
                assert obj.method_used == "propagation"

    def test_organ_name_is_label(self):
        """organ_name must equal the label from the reference frame."""
        result, _, _ = self._propagate([0, 1], n_targets=3)
        for labeled in result.values():
            for obj in labeled:
                assert obj.organ_name.startswith("cluster_")


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------

def test_bad_cluster_excluded_from_result():
    """Cluster failing min_image_frequency must not appear in the output."""
    labeled_by_image = _build_dataset(n_images=10, cluster_ids=[0])
    for i in range(2):
        path = Path(f"/tmp/img_{i}.png")
        labeled_by_image[path].append(
            _make_labeled(organ_id=1, source_path=str(path))
        )

    stub = StubVideoSegmenter()
    config = PropagationConfig(
        references_per_cluster=2,
        quality=ClusterQualityConfig(
            min_image_frequency=0.5,
            min_avg_labeling_confidence=0.0,
            min_avg_sam_confidence=0.0,
        ),
        mask_selection=MaskSelectionConfig(
            sam_score_weight=0.5, min_combined_score=0.0
        ),
    )
    result, _, _ = _build_and_propagate(
        labeled_by_image, config, [Path("/tmp/t.png")], stub
    )
    for labeled in result.values():
        organ_names = {obj.organ_name for obj in labeled}
        assert "cluster_1" not in organ_names, (
            f"Bad cluster_1 should be absent; got {organ_names}"
        )
        assert "cluster_0" in organ_names


def test_no_good_clusters_returns_empty():
    """When no cluster passes quality thresholds, result must be all-empty."""
    labeled_by_image = _build_dataset(
        n_images=5, cluster_ids=[0], sam_confidence=0.3
    )
    stub = StubVideoSegmenter()
    config = PropagationConfig(
        references_per_cluster=1,
        quality=ClusterQualityConfig(
            min_image_frequency=0.0,
            min_avg_labeling_confidence=0.0,
            min_avg_sam_confidence=0.9,
        ),
        mask_selection=MaskSelectionConfig(min_combined_score=0.0),
    )
    reader = StubReader()
    refs, scores = build_unsupervised_references(labeled_by_image, config, reader)
    assert refs == [] and scores == []

    propagator = PrototypePropagator(segmenter=stub, config=config)
    target_paths = [Path(f"/tmp/t_{i}.png") for i in range(3)]
    result, memory, debug = propagator.propagate(refs, scores, target_paths, reader)
    assert stub.multi_ref_call_count == 0, "segmenter must not be called when no references"
    assert all(v == [] for v in result.values())
    assert memory == []
    assert debug == []


# ---------------------------------------------------------------------------
# Memory composition
# ---------------------------------------------------------------------------

class TestMemoryComposition:

    def _memory(self, cluster_ids, k=2):
        labeled_by_image = _build_dataset(n_images=10, cluster_ids=cluster_ids)
        stub = StubVideoSegmenter()
        config = _permissive_config(k=k)
        _, memory, _ = _build_and_propagate(
            labeled_by_image, config, [Path("/tmp/t.png")], stub
        )
        return memory

    def test_required_fields_present(self):
        required = {
            "frame_idx", "obj_id", "label",
            "source_path", "mask", "combined_score", "area",
        }
        for entry in self._memory([0, 1, 2], k=2):
            assert required.issubset(entry.keys())

    def test_frame_idx_sequential_from_zero(self):
        """frame_idx restarts at 0 for each cluster and is sequential within it."""
        from collections import defaultdict
        memory = self._memory([0, 1], k=3)
        by_label: dict[str, list[int]] = defaultdict(list)
        for entry in memory:
            by_label[entry["label"]].append(entry["frame_idx"])
        for label, idxs in by_label.items():
            assert idxs == list(range(len(idxs))), (
                f"{label}: expected frame_idx 0..{len(idxs)-1}, got {idxs}"
            )

    def test_mask_is_ndarray(self):
        for entry in self._memory([0, 1]):
            assert isinstance(entry["mask"], np.ndarray)

    def test_area_matches_mask(self):
        for entry in self._memory([0, 1]):
            assert entry["area"] == int(entry["mask"].sum())

    def test_combined_score_is_float(self):
        for entry in self._memory([0, 1]):
            assert isinstance(entry["combined_score"], float)

    def test_length_equals_reference_frame_count(self):
        """len(memory) == total prototypes used (sum of min(k, n_protos) per cluster)."""
        reader = StubReader()
        labeled_by_image = _build_dataset(n_images=10, cluster_ids=[0, 1, 2])
        config = _permissive_config(k=2)
        stub = StubVideoSegmenter()
        refs, scores = build_unsupervised_references(labeled_by_image, config, reader)
        propagator = PrototypePropagator(segmenter=stub, config=config)
        _, memory, _ = propagator.propagate(refs, scores, [Path("/tmp/t.png")], reader)
        # mono-organ references: one memory entry per reference frame
        assert len(memory) == len(refs)


# ---------------------------------------------------------------------------
# Debug flag
# ---------------------------------------------------------------------------

def test_debug_flag_returns_empty_debug_when_false():
    """Third return element must be [] when debug_collect_reference_predictions=False."""
    labeled_by_image = _build_dataset(n_images=5, cluster_ids=[0, 1])
    stub = StubVideoSegmenter()
    config = _permissive_config(k=2)
    assert not config.debug_collect_reference_predictions
    _, _, debug = _build_and_propagate(
        labeled_by_image, config, [Path("/tmp/t.png")], stub
    )
    assert debug == []


# ---------------------------------------------------------------------------
# Iterative mode
# ---------------------------------------------------------------------------

def test_iterative_mode_calls_per_cluster_batch_once_per_cluster():
    """mode='iterative' calls segment_batch_iterative_per_cluster once per
    cluster, not segment_with_multi_reference, and produces the same output."""
    n_targets = 5
    n_clusters = 2
    labeled_by_image = _build_dataset(n_images=10, cluster_ids=[0, 1])
    stub = StubVideoSegmenter()
    config = PropagationConfig(
        references_per_cluster=2,
        quality=ClusterQualityConfig(
            min_image_frequency=0.0,
            min_avg_labeling_confidence=0.0,
            min_avg_sam_confidence=0.0,
        ),
        mask_selection=MaskSelectionConfig(
            sam_score_weight=0.5,
            min_combined_score=0.0,
        ),
        mode="iterative",
    )
    target_paths = [Path(f"/tmp/tgt_{i}.png") for i in range(n_targets)]
    result, _, _ = _build_and_propagate(labeled_by_image, config, target_paths, stub)

    assert stub.batch_per_cluster_call_count == n_clusters, (
        "segment_batch_iterative_per_cluster must be called once per cluster"
    )
    assert stub.multi_ref_call_count == 0, (
        "segment_with_multi_reference must not be called in iterative mode"
    )
    assert stub.batch_iterative_call_count == 0, (
        "segment_batch_iterative (old shared method) must not be called"
    )
    # Each per-cluster call spans all N targets
    for organ_name, n_refs, n_tgts in stub.batch_per_cluster_calls:
        assert n_tgts == n_targets, (
            f"{organ_name}: expected {n_targets} targets per session, got {n_tgts}"
        )
    assert set(result.keys()) == set(target_paths)
    expected_names = {"cluster_0", "cluster_1"}
    for path, labeled in result.items():
        assert {obj.organ_name for obj in labeled} == expected_names


# ---------------------------------------------------------------------------
# Few-shot direct propagation
# ---------------------------------------------------------------------------

def _make_fewshot_refs(organ_names: list[str]) -> list[FewShotReference]:
    """One mono-organ reference frame per organ name."""
    return [
        FewShotReference(
            volume=np.zeros((H, W, 3), dtype=np.float32),
            masks={name: np.ones((H, W), dtype=bool)},
            source_path=f"/tmp/ref_{i}.png",
        )
        for i, name in enumerate(organ_names)
    ]


class TestFewShotPropagation:

    def _propagate(self, organ_names, n_targets=3, propagation_mode="independent"):
        refs = _make_fewshot_refs(organ_names)
        fewshot_refs, scores = build_fewshot_references(refs)
        stub = StubVideoSegmenter()
        config = PropagationConfig(
            references_per_cluster=1,
            quality=ClusterQualityConfig(
                min_image_frequency=0.0,
                min_avg_labeling_confidence=0.0,
                min_avg_sam_confidence=0.0,
            ),
            mask_selection=MaskSelectionConfig(min_combined_score=0.0),
            mode=propagation_mode,
        )
        propagator = PrototypePropagator(segmenter=stub, config=config)
        target_paths = [Path(f"/tmp/tgt_{i}.png") for i in range(n_targets)]
        result, _, _ = propagator.propagate(fewshot_refs, scores, target_paths, StubReader())
        return result, stub

    def test_output_inherits_organ_names(self):
        """Propagated objects must carry the real organ names from references."""
        result, _ = self._propagate(["liver", "spleen"])
        for path, labeled in result.items():
            names = {obj.organ_name for obj in labeled}
            assert names == {"liver", "spleen"}, (
                f"Expected {{liver, spleen}}, got {names}"
            )

    def test_no_cluster_prefix_in_output(self):
        """Organ names must not start with 'cluster_' when using human references."""
        result, _ = self._propagate(["kidney", "liver"])
        for labeled in result.values():
            for obj in labeled:
                assert not obj.organ_name.startswith("cluster_"), (
                    f"Unexpected cluster name in few-shot output: {obj.organ_name}"
                )

    def test_frame_scores_are_one(self):
        """build_fewshot_references must assign score 1.0 to every frame."""
        refs = _make_fewshot_refs(["liver", "spleen", "kidney"])
        _, scores = build_fewshot_references(refs)
        assert scores == [1.0, 1.0, 1.0]

    def test_no_clustering_invoked(self):
        """method_used must be 'propagation', never a clustering method."""
        result, _ = self._propagate(["liver"])
        for labeled in result.values():
            for obj in labeled:
                assert obj.method_used == "propagation"

    def test_respects_iterative_propagation_mode(self):
        """few_shot propagation must call segment_batch_iterative_per_cluster
        once per organ when mode='iterative'."""
        organ_names = ["liver", "spleen"]
        result, stub = self._propagate(organ_names, n_targets=4,
                                       propagation_mode="iterative")
        assert stub.batch_per_cluster_call_count == len(organ_names), (
            "segment_batch_iterative_per_cluster must be called once per organ"
        )
        assert stub.multi_ref_call_count == 0, (
            "segment_with_multi_reference must not be called in iterative mode"
        )
        assert stub.batch_iterative_call_count == 0, (
            "old shared segment_batch_iterative must not be called"
        )
        for labeled in result.values():
            names = {obj.organ_name for obj in labeled}
            assert "liver" in names and "spleen" in names

    def test_multi_organ_reference_frame(self):
        """A single multi-organ reference frame must propagate all its organs."""
        multi_ref = FewShotReference(
            volume=np.zeros((H, W, 3), dtype=np.float32),
            masks={
                "liver": np.ones((H, W), dtype=bool),
                "spleen": np.ones((H, W), dtype=bool),
            },
            source_path="/tmp/ref_multi.png",
        )
        fewshot_refs, scores = build_fewshot_references([multi_ref])
        assert scores == [1.0]
        stub = StubVideoSegmenter()
        config = PropagationConfig(
            references_per_cluster=1,
            quality=ClusterQualityConfig(
                min_image_frequency=0.0,
                min_avg_labeling_confidence=0.0,
                min_avg_sam_confidence=0.0,
            ),
            mask_selection=MaskSelectionConfig(min_combined_score=0.0),
        )
        propagator = PrototypePropagator(segmenter=stub, config=config)
        result, _, _ = propagator.propagate(
            fewshot_refs, scores, [Path("/tmp/t.png")], StubReader()
        )
        for labeled in result.values():
            names = {obj.organ_name for obj in labeled}
            assert "liver" in names and "spleen" in names
