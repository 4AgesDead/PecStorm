"""Применить корректор центра к детекциям: сдвинуть боксы к GT-пивоту.

Читает видео покадрово, для каждой детекции с score >= 0.5 берёт кроп
(2× бокса), предсказывает смещение центра и сдвигает бокс. Результат — CSV
в том же формате, что и входной, для ``apps/reconstruct_truck.py``.

Запуск:
    .venv-rocm\\Scripts\\python.exe experiments/apply_center_refiner.py `
        --model out/experiment_kp/refiner.pt --out out/truck/det_center.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import torch
from torchvision import transforms

from center_model import CamResNet

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def crop_around(frame: np.ndarray, cx: float, cy: float, side: float,
                out_size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    half = side / 2.0
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = x0 + int(round(side)), y0 + int(round(side))
    left, top = max(0, -x0), max(0, -y0)
    right, bottom = max(0, x1 - w), max(0, y1 - h)
    crop = frame[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if left or top or right or bottom:
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right,
                                  cv2.BORDER_REPLICATE)
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def load_detections(path: Path) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(int(r["frame"]), []).append(r)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Сдвиг центров детекций к пивоту")
    ap.add_argument("--model", default="out/experiment_kp/refiner.pt")
    ap.add_argument("--detections", default="out/truck/det_ens_v4b_v2v1.csv")
    ap.add_argument("--out", default="out/truck/det_center.csv")
    ap.add_argument("--crop-scale", type=float, default=2.0)
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    cam_names = ckpt.get("cam_names") or ["cam1", "cam2", "cam3", "cam4", "cam5"]
    model = CamResNet(len(cam_names), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    by_frame = load_detections(Path(args.detections))
    out_rows: list[dict] = []
    moved = 0
    total = 0

    from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset

    ds = TruckDataset.discover(DEFAULT_ROOT, repairs_path=DEFAULT_REPAIRS)
    for cam in ds.cameras:
        cap = cv2.VideoCapture(str(cam.video))
        if not cap.isOpened():
            print(f"{cam.name}: не открыть видео", file=sys.stderr)
            continue
        f = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for det in by_frame.get(f, []):
                if det["camera"] != cam.name:
                    continue
                total += 1
                row = dict(det)
                score = float(det.get("score", 1.0))
                x1, y1, x2, y2 = (float(det[k]) for k in ("x1", "y1", "x2", "y2"))
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if score < args.min_score:
                    out_rows.append(row)
                    continue
                side = max(args.crop_scale * max(x2 - x1, y2 - y1), 64.0)
                crop = crop_around(frame, cx, cy, side, args.size)
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                x = TF(rgb).unsqueeze(0).to(device)
                cam_t = torch.tensor([cam_names.index(cam.name)],
                                     device=device, dtype=torch.long)
                with torch.no_grad():
                    pred = model(x, cam_t)[0].cpu().numpy()
                pred = np.clip(pred, -0.5, 0.5)
                dx, dy = float(pred[0] * side), float(pred[1] * side)
                shift = float(np.hypot(dx, dy))
                if shift > side * 0.35:
                    out_rows.append(row)
                    continue
                row["x1"] = f"{x1 + dx:.3f}"
                row["x2"] = f"{x2 + dx:.3f}"
                row["y1"] = f"{y1 + dy:.3f}"
                row["y2"] = f"{y2 + dy:.3f}"
                row["cx"] = f"{(cx + dx):.3f}"
                row["cy"] = f"{(cy + dy):.3f}"
                out_rows.append(row)
                moved += 1
            f += 1
        cap.release()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(out_rows[0].keys()) if out_rows else []
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"детекций: {total}, сдвинуто: {moved} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
