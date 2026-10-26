import torch

from s2cd.metrics import ChangeMetrics


def test_perfect_prediction():
    m = ChangeMetrics()
    logits = torch.tensor([[[10.0, -10.0]], [[-10.0, 10.0]]])  # (2,2,1) logits
    target = torch.tensor([[0], [1]])
    m.update(logits, target)
    r = m.results()
    assert r["miou"] == 1.0
    assert r["change_f1"] == 1.0
    assert r["pixel_acc"] == 1.0


def test_known_confusion():
    m = ChangeMetrics()
    # 4 pixels: 2 correct change, 1 false positive, 1 missed change
    target = torch.tensor([[1], [1], [0], [1]])
    # logits honoring predictions: change for the first 3, no-change for the last
    logits = torch.stack([
        torch.tensor([[0.0, 10.0]]),
        torch.tensor([[0.0, 10.0]]),
        torch.tensor([[0.0, 10.0]]),
        torch.tensor([[10.0, 0.0]]),
    ])
    m.update(logits, target)
    r = m.results()
    assert abs(r["change_precision"] - 2 / 3) < 1e-6
    assert abs(r["change_recall"] - 2 / 3) < 1e-6
    assert abs(r["change_f1"] - 2 / 3) < 1e-6
    assert r["confusion"][1][1] == 2 and r["confusion"][0][1] == 1 and r["confusion"][1][0] == 1