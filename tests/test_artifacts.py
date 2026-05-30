"""Round-trip tests for Phase 1 segmentation artifacts."""

from pathlib import Path

import numpy as np
import pytest

from project.core.data_types import MedicalImage, SegmentedObject
from project.pipeline.artifacts import load_segmented, save_segmented, write_cache_key


# ── Helpers ────────────────────────────────────────────────────────────────

def _med_image(source_path: str, modality: str = "xray", metadata: dict | None = None) -> MedicalImage:
    return MedicalImage(
        volume=None,
        modality=modality,
        metadata=metadata or {},
        source_path=source_path,
    )


def _make_obj(
    img: MedicalImage,
    mask_shape: tuple = (50, 50),
    with_embedding: bool = True,
    confidence: float = 0.9,
    label: str | None = "organ_a",
) -> SegmentedObject:
    mask = np.zeros(mask_shape, dtype=bool)
    mask[5:20, 5:20] = True
    features = np.arange(10, dtype=np.float32) * 0.1
    embedding = np.random.default_rng(0).random(256).astype(np.float32) if with_embedding else None
    return SegmentedObject(
        mask=mask,
        source_image=img,
        features=features,
        embedding=embedding,
        confidence=confidence,
        label=label,
    )


# ── Round-trip tests ───────────────────────────────────────────────────────

def test_round_trip_basic(tmp_path):
    img1 = _med_image("/data/img001.png", modality="xray", metadata={"patient": "001"})
    img2 = _med_image("/data/img002.png", modality="ct")

    obj1 = _make_obj(img1)
    obj2 = _make_obj(img1, confidence=0.75, label="organ_b")
    obj3 = _make_obj(img2, with_embedding=False)

    original_ids = [obj1.id, obj2.id, obj3.id]

    objects_by_image = {
        Path("/data/img001.png"): [obj1, obj2],
        Path("/data/img002.png"): [obj3],
    }

    out_dir = tmp_path / "seg_cache"
    save_segmented(
        objects_by_image,
        out_dir,
        hash8="abcd1234",
        dataset_short="TestDS",
        mode="unsupervised",
        propagation_mode="-",
    )

    all_loaded, loaded_by_image = load_segmented(out_dir)

    # Total count
    assert len(all_loaded) == 3

    # IDs restored exactly, in the same order
    loaded_ids = [o.id for o in all_loaded]
    assert loaded_ids == original_ids


def test_round_trip_grouping(tmp_path):
    img1 = _med_image("/data/img001.png")
    img2 = _med_image("/data/img002.png")
    objects_by_image = {
        Path("/data/img001.png"): [_make_obj(img1), _make_obj(img1)],
        Path("/data/img002.png"): [_make_obj(img2)],
    }
    out_dir = tmp_path / "seg"
    save_segmented(objects_by_image, out_dir, "ab12ef34", "DS", "unsupervised", "-")
    _, loaded = load_segmented(out_dir)

    assert set(loaded.keys()) == {Path("/data/img001.png"), Path("/data/img002.png")}
    assert len(loaded[Path("/data/img001.png")]) == 2
    assert len(loaded[Path("/data/img002.png")]) == 1


def test_masks_exact(tmp_path):
    img = _med_image("/data/img.png")
    obj = _make_obj(img)
    objects_by_image = {Path("/data/img.png"): [obj]}
    out_dir = tmp_path / "seg"
    save_segmented(objects_by_image, out_dir, "00000000", "DS", "unsupervised", "-")
    _, loaded = load_segmented(out_dir)

    loaded_obj = loaded[Path("/data/img.png")][0]
    np.testing.assert_array_equal(obj.mask, loaded_obj.mask)


def test_features_exact(tmp_path):
    img = _med_image("/data/img.png")
    obj = _make_obj(img)
    objects_by_image = {Path("/data/img.png"): [obj]}
    out_dir = tmp_path / "seg"
    save_segmented(objects_by_image, out_dir, "00000000", "DS", "unsupervised", "-")
    _, loaded = load_segmented(out_dir)

    loaded_obj = loaded[Path("/data/img.png")][0]
    np.testing.assert_array_equal(obj.features, loaded_obj.features)


def test_embedding_exact(tmp_path):
    img = _med_image("/data/img.png")
    obj = _make_obj(img, with_embedding=True)
    objects_by_image = {Path("/data/img.png"): [obj]}
    out_dir = tmp_path / "seg"
    save_segmented(objects_by_image, out_dir, "00000000", "DS", "unsupervised", "-")
    _, loaded = load_segmented(out_dir)

    loaded_obj = loaded[Path("/data/img.png")][0]
    assert loaded_obj.embedding is not None
    np.testing.assert_array_equal(obj.embedding, loaded_obj.embedding)


def test_embedding_none_preserved(tmp_path):
    img = _med_image("/data/img.png")
    obj = _make_obj(img, with_embedding=False)
    objects_by_image = {Path("/data/img.png"): [obj]}
    out_dir = tmp_path / "seg"
    save_segmented(objects_by_image, out_dir, "00000000", "DS", "unsupervised", "-")
    _, loaded = load_segmented(out_dir)

    assert loaded[Path("/data/img.png")][0].embedding is None


def test_confidence_and_label(tmp_path):
    img = _med_image("/data/img.png")
    obj = _make_obj(img, confidence=0.77, label="liver")
    objects_by_image = {Path("/data/img.png"): [obj]}
    out_dir = tmp_path / "seg"
    save_segmented(objects_by_image, out_dir, "00000000", "DS", "unsupervised", "-")
    _, loaded = load_segmented(out_dir)

    lo = loaded[Path("/data/img.png")][0]
    assert lo.confidence == pytest.approx(0.77)
    assert lo.label == "liver"


def test_source_image_fields(tmp_path):
    img = _med_image("/data/img.png", modality="ct", metadata={"key": "val"})
    obj = _make_obj(img)
    objects_by_image = {Path("/data/img.png"): [obj]}
    out_dir = tmp_path / "seg"
    save_segmented(objects_by_image, out_dir, "00000000", "DS", "unsupervised", "-")
    _, loaded = load_segmented(out_dir)

    lo = loaded[Path("/data/img.png")][0]
    assert lo.source_image.source_path == "/data/img.png"
    assert lo.source_image.modality == "ct"
    assert lo.source_image.metadata == {"key": "val"}
    assert lo.source_image.volume is None


def test_label_none(tmp_path):
    img = _med_image("/data/img.png")
    obj = _make_obj(img, label=None)
    objects_by_image = {Path("/data/img.png"): [obj]}
    out_dir = tmp_path / "seg"
    save_segmented(objects_by_image, out_dir, "00000000", "DS", "unsupervised", "-")
    _, loaded = load_segmented(out_dir)

    assert loaded[Path("/data/img.png")][0].label is None


def test_shared_med_image_per_image(tmp_path):
    """All objects from the same image share the same MedicalImage instance."""
    img = _med_image("/data/img.png")
    obj1 = _make_obj(img)
    obj2 = _make_obj(img)
    objects_by_image = {Path("/data/img.png"): [obj1, obj2]}
    out_dir = tmp_path / "seg"
    save_segmented(objects_by_image, out_dir, "00000000", "DS", "unsupervised", "-")
    _, loaded = load_segmented(out_dir)

    a, b = loaded[Path("/data/img.png")]
    assert a.source_image is b.source_image


def test_npz_is_created(tmp_path):
    img = _med_image("/data/img.png")
    objects_by_image = {Path("/data/img.png"): [_make_obj(img)]}
    out_dir = tmp_path / "seg"
    save_segmented(objects_by_image, out_dir, "00000000", "DS", "unsupervised", "-")
    assert (out_dir / "segmented.npz").exists()
    assert (out_dir / "segmented.json").exists()


def test_write_cache_key(tmp_path):
    write_cache_key(tmp_path, {"key": "value"}, "abcd1234")
    import json
    data = json.loads((tmp_path / "cache_key.json").read_text())
    assert data["hash8"] == "abcd1234"
    assert data["payload"] == {"key": "value"}
    assert "created_at" in data


def test_id_not_regenerated(tmp_path):
    """Critical: loaded object IDs must match originals exactly."""
    img = _med_image("/data/img.png")
    objs = [_make_obj(img) for _ in range(5)]
    original_ids = [o.id for o in objs]
    objects_by_image = {Path("/data/img.png"): objs}
    out_dir = tmp_path / "seg"
    save_segmented(objects_by_image, out_dir, "00000000", "DS", "unsupervised", "-")
    all_loaded, _ = load_segmented(out_dir)
    assert [o.id for o in all_loaded] == original_ids
