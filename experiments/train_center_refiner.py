"""Пилот: ResNet18 предсказывает смещение GT-пивота от центра кропа.

Вход — кроп вокруг детекции (2× бокса), выход — (dx, dy) в долях стороны кропа.
Loss — L1 в пикселях, чтобы оптимизировать именно ошибку центра.

Запуск:
    .venv-rocm\\Scripts\\python.exe experiments/train_center_refiner.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from center_model import CamResNet

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class Crops(Dataset):
    def __init__(self, rows: list[dict], train: bool, cam_names: list[str]):
        self.rows = rows
        self.cam_index = {n: i for i, n in enumerate(cam_names)}
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
            if train else transforms.Lambda(lambda im: im),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(r["path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.tf(img)
        y = torch.tensor([float(r["label_x"]), float(r["label_y"])],
                         dtype=torch.float32)
        side = torch.tensor(float(r["side"]), dtype=torch.float32)
        cam = torch.tensor(self.cam_index[r["camera"]], dtype=torch.long)
        return x, y, side, cam


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    errs = []
    for x, y, side, cam in loader:
        x, y, side, cam = (x.to(device), y.to(device), side.to(device),
                           cam.to(device))
        with torch.set_grad_enabled(train):
            pred = model(x, cam)
            err_px = (pred - y) * side.unsqueeze(1)
            loss = err_px.abs().mean()
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += float(loss) * x.size(0)
        n += x.size(0)
        errs.append(err_px.detach().cpu().norm(dim=1))
    e = torch.cat(errs).numpy()
    return total / max(n, 1), float(np.median(e)), float(np.mean(e))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Корректор центра бокса (пилот)")
    ap.add_argument("--samples", default="out/experiment_kp/samples.csv")
    ap.add_argument("--out", default="out/experiment_kp/refiner.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    rows = read_rows(Path(args.samples))
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val_gap"]
    hold_rows = [r for r in rows if r["split"] == "holdout"]
    print(f"train={len(train_rows)} val_gap={len(val_rows)} holdout={len(hold_rows)}")
    for name, rs in (("train", train_rows), ("val_gap", val_rows),
                     ("holdout", hold_rows)):
        if rs:
            base = np.array([float(r["baseline_px"]) for r in rs])
            print(f"  {name}: baseline центр, медиана {np.median(base):.1f} px, "
                  f"P90 {np.percentile(base, 90):.1f}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cam_names = sorted({r["camera"] for r in rows})
    model = CamResNet(len(cam_names), pretrained=True)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    train_loader = DataLoader(Crops(train_rows, True, cam_names),
                              batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(Crops(val_rows, False, cam_names),
                            batch_size=args.batch, shuffle=False,
                            num_workers=0, pin_memory=True)
    hold_loader = DataLoader(Crops(hold_rows, False, cam_names),
                             batch_size=args.batch, shuffle=False,
                             num_workers=0, pin_memory=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_med, tr_mean = run_epoch(model, train_loader, device, opt)
        va_loss, va_med, va_mean = run_epoch(model, val_loader, device)
        sched.step()
        print(f"epoch {epoch:2d}: train L1={tr_loss:6.2f}px med={tr_med:5.1f} | "
              f"val L1={va_loss:6.2f}px med={va_med:5.1f} mean={va_mean:5.1f}")
        history.append({"epoch": epoch, "train_l1": tr_loss,
                        "train_median": tr_med, "val_l1": va_loss,
                        "val_median": va_med, "val_mean": va_mean})
        if va_med < best:
            best = va_med
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_median_px": va_med, "args": vars(args),
                        "cam_names": cam_names}, out)

    ckpt = torch.load(out, map_location=device)
    model = CamResNet(len(ckpt["cam_names"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    hv_loss, hv_med, hv_mean = run_epoch(model, hold_loader, device)
    print(f"\nbest epoch {ckpt['epoch']}: val median {ckpt['val_median_px']:.1f} px")
    print(f"holdout: L1={hv_loss:.2f}px median={hv_med:.1f} mean={hv_mean:.1f}")
    (out.parent / "refiner_history.json").write_text(
        json.dumps({"history": history, "holdout": {"l1": hv_loss, "median": hv_med,
                                                    "mean": hv_mean}}, indent=2),
        encoding="utf-8")
    print(f"чекпойнт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
