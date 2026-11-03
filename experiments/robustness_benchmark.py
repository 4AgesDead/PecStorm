"""Робастность геометрии: шум 2D, ошибки калибровки, рассогласование времени.

Запуск:
    python experiments/robustness_benchmark.py [--out out/experiments] [--seed 7]

Сцена — синтетика ``make_sim_setup`` с известной истинной траекторией, поэтому
измеряется **истинная 3D-ошибка**, а не только репроекционная невязка (§7.4):
* шум 2D-точек 0 / 0.25 / 0.5 / 1 / 2 / 5 px;
* ошибка focal length ±1…2 % и сдвиг principal point 5…10 px;
* ошибка поворота камеры 0.5…2°;
* рассогласование метки времени 2…50 мс (ошибка = путь за это время).

Отчёт: ``out/experiments/robustness.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from multiview3d.camera import project_point
from multiview3d.fusion import FusionConfig, MultiViewTracker
from multiview3d.sim import make_sim_setup
from multiview3d.types import Detection2D

CFG = FusionConfig(association_mode="single_target", max_reprojection_px=100.0)


def triangulate(cameras, truth: np.ndarray, t: float, noise_px: float,
                rng: np.random.Generator, gen_cameras=None) -> np.ndarray | None:
    """Пиксели от ``gen_cameras`` (истинная модель), триангуляция — ``cameras``."""
    tracker = MultiViewTracker([f"c{i}" for i in range(len(cameras))], cameras, CFG)
    gen = cameras if gen_cameras is None else gen_cameras
    dets = []
    for i, cam in enumerate(cameras):
        uv = project_point(gen[i], truth)
        if noise_px:
            uv = uv + rng.normal(0.0, noise_px, size=2)
        dets.append([Detection2D(cam.name, t, uv, 1.0)])
    res = tracker.step(t, dets)
    if not res.candidates:
        return None
    return np.asarray(res.candidates[0].position, dtype=float)


def benchmark_noise(ds, seeds=(1, 2, 3)) -> list[dict]:
    out = []
    for noise in (0.0, 0.25, 0.5, 1.0, 2.0, 5.0):
        errs = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for t in ds.frame_times:
                truth = np.asarray(ds.paths[0](t), dtype=float)
                est = triangulate(ds.cameras, truth, t, noise, rng)
                if est is not None:
                    errs.append(float(np.linalg.norm(est - truth)))
        a = np.asarray(errs)
        out.append({"noise_px": noise, "n": int(a.size),
                    "mean_m": round(float(a.mean()), 4) if a.size else None,
                    "median_m": round(float(np.median(a)), 4) if a.size else None,
                    "max_m": round(float(a.max()), 4) if a.size else None,
                    "rejected": int(len(ds.frame_times) * len(seeds) - a.size)})
    return out


def perturb_calibration(cameras, *, focal_scale=1.0, pp_shift_px=0.0,
                        rot_deg=0.0) -> list:
    out = []
    for cam in cameras:
        c = copy.deepcopy(cam)
        if focal_scale != 1.0:
            c.K[0, 0] *= focal_scale
            c.K[1, 1] *= focal_scale
            c.K[0, 2] = cam.width / 2.0
            c.K[1, 2] = cam.height / 2.0
        if pp_shift_px:
            c.K[0, 2] += pp_shift_px
            c.K[1, 2] += pp_shift_px
        if rot_deg:
            a = np.radians(rot_deg)
            rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)],
                           [0, np.sin(a), np.cos(a)]])
            c.R = rx @ c.R
        out.append(c)
    return out


def benchmark_calibration(ds) -> list[dict]:
    cases = [
        ("focal +1%", {"focal_scale": 1.01}),
        ("focal +2%", {"focal_scale": 1.02}),
        ("principal point +5px", {"pp_shift_px": 4.0}),
        ("principal point +10px", {"pp_shift_px": 8.0}),
        ("rotation 0.5°", {"rot_deg": 0.5}),
        ("rotation 2°", {"rot_deg": 2.0}),
    ]
    rng = np.random.default_rng(0)
    out = []
    for name, kwargs in cases:
        cams = perturb_calibration(ds.cameras, **kwargs)
        errs = [np.linalg.norm(triangulate(cams, np.asarray(ds.paths[0](t), float),
                                           t, 0.0, rng, gen_cameras=ds.cameras)
                               - np.asarray(ds.paths[0](t), float))
                for t in ds.frame_times]
        a = np.asarray([e for e in errs if np.isfinite(e)])
        out.append({"case": name, **kwargs, "n": int(a.size),
                    "mean_m": round(float(a.mean()), 4) if a.size else None,
                    "max_m": round(float(a.max()), 4) if a.size else None})
    return out


def benchmark_timing(ds) -> list[dict]:
    """Ошибка позиции, если детекции приписаны неверному моменту времени."""
    times = np.asarray(ds.frame_times)
    pos = np.asarray([ds.paths[0](t) for t in times], dtype=float)
    vel = np.gradient(pos, times, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    out = []
    for ms in (0, 1, 2, 5, 10, 20, 50):
        dt = ms / 1000.0
        if dt == 0:
            err = 0.0
        else:
            shifted = np.asarray([ds.paths[0](min(t + dt, times[-1])) for t in times], float)
            err = float(np.mean(np.linalg.norm(shifted - pos, axis=1)))
        out.append({"offset_ms": ms, "mean_error_m": round(err, 4),
                    "mean_speed_m_s": round(float(speed.mean()), 3)})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Робастность геометрии (синтетика)")
    ap.add_argument("--out", default="out/experiments")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    ds = make_sim_setup(num_cameras=3, num_objects=1, fps=25,
                        duration_s=1.0, seed=args.seed)
    noise = benchmark_noise(ds)
    calib = benchmark_calibration(ds)
    timing = benchmark_timing(ds)

    print("Шум 2D (истинная 3D-ошибка, м):")
    for r in noise:
        print(f"  {r['noise_px']:>4} px: mean={r['mean_m']}, max={r['max_m']}, "
              f"отброшено={r['rejected']}")
    print("Ошибки калибровки:")
    for r in calib:
        print(f"  {r['case']:<24}: mean={r['mean_m']} м, max={r['max_m']} м")
    print("Рассогласование времени:")
    for r in timing:
        print(f"  {r['offset_ms']:>3} мс: mean={r['mean_error_m']} м "
              f"(скорость {r['mean_speed_m_s']} м/с)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"sim": {"cameras": 3, "objects": 1, "fps": 25, "duration_s": 1.0,
                   "seed": args.seed},
           "note": ("Истинная 3D-ошибка на известной синтетической траектории; "
                    "величины зависят от геометрии сцены, это не оценка датасета."),
           "noise": noise, "calibration": calib, "timing": timing}
    (out / "robustness.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"JSON: {out / 'robustness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
