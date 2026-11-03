"""Визуализация оценок 3D на кадрах датасета «Пеленга».

Запуск:
    python apps/visualize_truck.py --estimates out/truck/estimates.csv
        --frames 0,200,400,600,800,1000 --out out/truck/viz [--video]

Для каждого выбранного кадра и каждой камеры:
* зелёный крест — проекция **оценки** конвейера (цвет зависит от статуса:
  зелёный = MEASURED, оранжевый = PREDICTED_ONLY);
* жёлтый крест — проекция GT-пивота (только для контроля, в оценке не
  используется);
* подпись: номер кадра, имя камеры, статус, текстом — ошибка в метрах.

GT читается через ``TruckDataset`` (битые ячейки исключены). Оценки — CSV из
``apps/reconstruct_truck.py``. Исходные видео не изменяются.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

from multiview3d.camera import project_point
from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset
from multiview3d.truck_pipeline import load_estimates
from multiview3d.types import MEASURED

COLORS = {MEASURED: (0, 255, 0)}


def read_frame(path: Path, frame: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def draw(img: np.ndarray, uv, color, label: str, size: int = 34,
         text_y: int = 34) -> bool:
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    inside = 0 <= u < img.shape[1] and 0 <= v < img.shape[0]
    cv2.drawMarker(img, (u, v), color, cv2.MARKER_CROSS, size, 3)
    cv2.putText(img, label + ("" if inside else " OFF-SCREEN"),
                (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return inside


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Визуализация оценок 3D на кадрах «Пеленги»")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repairs", default=str(DEFAULT_REPAIRS))
    ap.add_argument("--estimates", required=True)
    ap.add_argument("--frames", default="0,200,400,600,800,1000")
    ap.add_argument("--out", default="out/truck/viz")
    ap.add_argument("--video", action="store_true",
                    help="дописать montage PNG (сетка кадров камер)")
    ap.add_argument("--fps", type=float, default=None)
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
    pts = load_estimates(args.estimates)
    by_frame: dict[int, object] = {}
    for p in pts:
        f = p.frame if p.frame is not None else int(round(p.timestamp * ds.fps))
        cur = by_frame.get(f)
        if cur is None or (p.status == MEASURED and cur.status != MEASURED):
            by_frame[f] = p
    gt = ds.ground_truth()
    gt_by_frame = {int(f): p for f, p in zip(gt.frames, gt.positions_m)}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [int(x) for x in args.frames.split(",") if x.strip()]
    summary = []
    for f in frames:
        p = by_frame.get(f)
        cams = ds.provider.calibrations_at(f, nearest=True)
        written = []
        for cam_name in ds.names:
            cam = cams.get(cam_name)
            video = next(c.video for c in ds.cameras if c.name == cam_name)
            img = read_frame(video, f)
            if img is None or cam is None:
                continue
            row = {"frame": f, "camera": cam_name}
            if p is not None:
                color = COLORS.get(p.status, (0, 165, 255))
                inside = draw(img, project_point(cam, p.position), color,
                              f"{cam_name} f={f} {p.status}")
                row.update({"status": p.status, "method": p.method,
                            "views": list(p.views), "inside": bool(inside)})
                if f in gt_by_frame:
                    err = float(np.linalg.norm(np.asarray(p.position) - gt_by_frame[f]))
                    row["error_m"] = round(err, 3)
                    cv2.putText(img, f"err={err:.2f} m", (10, 114),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if f in gt_by_frame:
                draw(img, project_point(cam, gt_by_frame[f]), (0, 255, 255),
                     f"GT f={f}", size=26, text_y=74)
            path = out_dir / f"{cam_name}_f{f:04d}.png"
            cv2.imwrite(str(path), img)
            written.append(path.name)
            summary.append(row)
            print(f"  {path}")
        if args.video and written:
            first = cv2.imread(str(out_dir / written[0]))
            tiles = [cv2.resize(cv2.imread(str(out_dir / w)), (first.shape[1] // 2,
                                                              first.shape[0] // 2))
                     for w in written]
            grid = np.vstack(tiles)
            cv2.imwrite(str(out_dir / f"montage_f{f:04d}.png"), grid)

    (out_dir / "viz_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    n_inside = sum(1 for r in summary if r.get("inside"))
    print(f"Кадров: {len(frames)}, картинок: {len(summary)}, "
          f"проекций в кадре: {n_inside}; сводка: {out_dir / 'viz_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
