"""Projection sanity: GT-пивот цели -> пиксели кадров, оверлеи и карта видимости.

Диагностика разработки (GT используется только для проверки трансформации координат,
не для предсказаний). Запуск:

    python apps/projection_sanity.py [--frames 0,150,300,...] [--step 30]

Создаёт: out/truck_sanity/overlay_cam<N>_frame<F>.png, visibility.json, summary.txt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from multiview3d.camera_provider import CameraSpec, CameraStateProvider  # noqa: E402
from multiview3d.camera import project_point  # noqa: E402
from multiview3d.unreal_camera import load_ground_truth  # noqa: E402

DATA_ROOT = Path("scene1/zadania")
GT_PATH = DATA_ROOT / "xyzpowcameta/object_transform_SM_AA_track.csv"
REPAIRS = Path("datasets/truck/ground_truth_repairs.json")


def build_provider(width: int = 1920, height: int = 1080) -> CameraStateProvider:
    specs = []
    for i in range(1, 6):
        csv_path = DATA_ROOT / f"xyzpowcameta/object_transform_CineCameraActor{i}.csv"
        video = DATA_ROOT / f"camera/Way1.FinalImage.CineCameraActor{i}.mp4"
        specs.append(CameraSpec(name=f"cam{i}", csv_path=str(csv_path),
                                width=width, height=height))
        assert video.exists(), video
    return CameraStateProvider(specs)


def video_path(cam_name: str) -> Path:
    i = cam_name.replace("cam", "")
    return DATA_ROOT / f"camera/Way1.FinalImage.CineCameraActor{i}.mp4"


def read_frame(cam_name: str, frame: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path(cam_name)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def project_pivot(cam, pos_m: np.ndarray) -> tuple[np.ndarray, float]:
    """Пиксель пивота + расстояние до центра камеры (м)."""
    uv = project_point(cam, pos_m)
    center = -cam.R.T @ cam.t
    return uv, float(np.linalg.norm(pos_m - center))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="0,150,300,450,600,750,900")
    ap.add_argument("--step", type=int, default=30, help="шаг карты видимости")
    ap.add_argument("--out", default="out/truck_sanity")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = build_provider()
    gt = load_ground_truth(GT_PATH, repairs_path=REPAIRS, strict=False)
    gt_pos = {int(f): p for f, p in zip(gt.frames, gt.positions_m)}

    # --- карта видимости: пивот в кадре (с запасом 0 px) по каждому кадру/шагу ---
    visibility: dict[int, dict[str, list[float]]] = {}
    frames_to_check = list(range(0, 1001, args.step))
    for f in frames_to_check:
        if f not in gt_pos:
            continue
        cams = provider.calibrations_at(f)
        row: dict[str, list[float]] = {}
        for name, cam in cams.items():
            try:
                uv, dist = project_pivot(cam, gt_pos[f])
            except ValueError:
                continue
            if 0 <= uv[0] < cam.width and 0 <= uv[1] < cam.height:
                row[name] = [round(float(uv[0]), 1), round(float(uv[1]), 1), round(dist, 2)]
        visibility[f] = row
    multi = [f for f, row in visibility.items() if len(row) >= 2]
    single = [f for f, row in visibility.items() if len(row) == 1]
    none = [f for f, row in visibility.items() if len(row) == 0]
    print(f"Карта видимости (шаг {args.step} кадр, всего {len(visibility)}):")
    for f in frames_to_check:
        row = visibility.get(f, {})
        print(f"  frame {f:4d}: {sorted(row.keys()) or '—'}"
              + (f"  dist={row[list(row)[0]][2]}m" if row else ""))
    print(f"кадров с >=2 камерами: {len(multi)}, с 1 камерой: {len(single)}, без камер: {len(none)}")
    (out_dir / "visibility.json").write_text(
        json.dumps({str(k): v for k, v in visibility.items()}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    # --- оверлеи для выбранных кадров ---
    for f in [int(x) for x in args.frames.split(",")]:
        row = visibility.get(f)
        cams = provider.calibrations_at(f)
        for name in cams:
            img = read_frame(name, f)
            if img is None:
                continue
            cam = cams[name]
            try:
                uv, dist = project_pivot(cam, gt_pos[f])
            except ValueError:
                continue
            u, v = int(round(uv[0])), int(round(uv[1]))
            inside = 0 <= u < img.shape[1] and 0 <= v < img.shape[0]
            color = (0, 255, 0) if inside else (0, 0, 255)
            cv2.drawMarker(img, (u, v), color, cv2.MARKER_CROSS, 40, 3)
            cv2.putText(img, f"{name} f={f} GT pivot ({u},{v}) {'' if inside else 'OFF-SCREEN'} "
                             f"{dist:.1f}m", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            p = out_dir / f"overlay_{name}_frame{f}.png"
            cv2.imwrite(str(p), img)
            print(f"оверлей: {p} ({'в кадре' if inside else 'вне кадра'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
