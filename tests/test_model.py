import pytest
import torch

from s2cd.models import SiameseUNet


def test_output_shape_matches_input():
    model = SiameseUNet(in_channels=10, base=32)
    t0 = torch.randn(2, 10, 64, 64)
    t1 = torch.randn(2, 10, 64, 64)
    out = model(t0, t1)
    assert out.shape == (2, 2, 64, 64), out.shape


def test_per_pixel_output_not_collapsed():
    """The V1 bug-class regression: output must vary spatially."""
    model = SiameseUNet(in_channels=10, base=32).eval()
    t0 = torch.zeros(1, 10, 64, 64)
    t1 = torch.zeros(1, 10, 64, 64)
    t1[0, 0, :8, :8] = 5.0  # change in one corner only
    with torch.no_grad():
        out = model(t0, t1)
    var = out[0, 1].std().item()
    assert var > 1e-6, "output is spatially constant — head collapsed"


def test_shared_encoder_symmetry():
    model = SiameseUNet(in_channels=10, base=32).eval()
    x = torch.randn(1, 10, 64, 64)
    with torch.no_grad():
        a = model(x, x)
        b = model(x, torch.zeros_like(x))
    assert not torch.allclose(a, b), "identical inputs must not differ from changed inputs"


def test_param_count_reasonable():
    model = SiameseUNet(in_channels=10, base=64)
    n = model.count_params()
    assert 1_000_000 < n < 50_000_000, n