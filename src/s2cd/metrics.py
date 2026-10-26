"""Segmentation metrics on a 2-class confusion matrix (background / change)."""

import numpy as np
import torch


class ChangeMetrics:
    def __init__(self):
        self.cm = np.zeros((2, 2), dtype=np.int64)  # [true, pred]

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor):
        pred = logits.argmax(1).cpu().numpy().ravel()
        tgt = target.cpu().numpy().ravel()
        self.cm += np.bincount(2 * tgt.astype(np.int64) + pred,
                                minlength=4).reshape(2, 2)

    def results(self) -> dict:
        tn, fp = self.cm[0, 0], self.cm[0, 1]
        fn, tp = self.cm[1, 0], self.cm[1, 1]
        change_iou = tp / max(tp + fp + fn, 1)
        bg_iou = tn / max(tn + fp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        acc = (tn + tp) / max(self.cm.sum(), 1)
        return {
            "miou": float((change_iou + bg_iou) / 2),
            "change_iou": float(change_iou),
            "change_precision": float(prec),
            "change_recall": float(rec),
            "change_f1": float(f1),
            "pixel_acc": float(acc),
            "confusion": self.cm.tolist(),
        }

    def summary(self) -> str:
        r = self.results()
        return (f"mIoU {r['miou']:.4f} | change IoU {r['change_iou']:.4f} | "
                f"F1 {r['change_f1']:.4f} | P {r['change_precision']:.4f} | "
                f"R {r['change_recall']:.4f} | pix-acc {r['pixel_acc']:.4f}")