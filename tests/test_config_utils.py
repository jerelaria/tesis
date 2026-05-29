"""Tests for config_utils: _parse_value and apply_config_overrides."""
import pytest
from project.core.config_utils import _parse_value, apply_config_overrides


def test_parse_true():
    assert _parse_value("true") is True
    assert _parse_value("True") is True
    assert _parse_value("TRUE") is True


def test_parse_false():
    assert _parse_value("false") is False


def test_parse_none():
    assert _parse_value("none") is None
    assert _parse_value("null") is None


def test_parse_int():
    assert _parse_value("42") == 42
    assert isinstance(_parse_value("42"), int)


def test_parse_float():
    result = _parse_value("3.14")
    assert abs(result - 3.14) < 1e-9
    assert isinstance(result, float)


def test_parse_list():
    result = _parse_value("a,b,c")
    assert result == ["a", "b", "c"]


def test_parse_list_mixed():
    result = _parse_value("1,2.5,true")
    assert result == [1, 2.5, True]


def test_parse_string_fallback():
    assert _parse_value("hello") == "hello"


def test_apply_overrides_nested():
    """Overrides with dot-notation should create nested keys."""
    cfg = {}
    apply_config_overrides(cfg, ["labeler.kmeans.n_clusters=5"])
    assert cfg["labeler"]["kmeans"]["n_clusters"] == 5


def test_apply_overrides_new_key():
    """Overrides can add entirely new nested keys."""
    cfg = {"a": {}}
    apply_config_overrides(cfg, ["a.b.c=42"])
    assert cfg["a"]["b"]["c"] == 42


def test_apply_overrides_invalid_format():
    """An override without '=' must raise ValueError."""
    with pytest.raises(ValueError, match="KEY=VALUE"):
        apply_config_overrides({}, ["no_equals_sign"])
