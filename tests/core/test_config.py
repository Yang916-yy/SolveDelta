import pytest

from causallsso import SolveDeltaConfig


def test_key_width_is_resolved_not_independently_ranked() -> None:
    direct = SolveDeltaConfig(2048, 16, head_k_dim=128)
    derived = SolveDeltaConfig(2048, 16, expand_k=1.0)
    expanded = SolveDeltaConfig(2048, 16, expand_k=2.0)

    assert direct.geometry_width == direct.resolved_head_k_dim == 128
    assert derived.geometry_width == 128
    assert expanded.geometry_width == 256
    assert expanded.key_dim == 4096
    assert direct.num_edits == 1
    assert direct.use_short_conv


def test_invalid_width_resolution_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        SolveDeltaConfig(10, 3)
    with pytest.raises(ValueError, match="must be an integer"):
        SolveDeltaConfig(10, 2, expand_k=1.25)
    with pytest.raises(ValueError, match="must be positive"):
        SolveDeltaConfig(16, 2, head_k_dim=0)
    with pytest.raises(TypeError, match="use_short_conv must be a bool"):
        SolveDeltaConfig(16, 2, use_short_conv=1)
