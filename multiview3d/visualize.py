"""Визуализация и экспорт: 3D-графики траекторий, наложение на кадр, CSV/JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .camera import project_point
from .types import CameraCalibration, Track


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")  # без экрана: файл на диск
    import matplotlib.pyplot as plt
    return plt


def plot_tracks_3d(tracks: list[Track],
                   truth_points: list[list[np.ndarray]] | None = None,
                   path: str | Path | None = None,
                   title: str = "Траектории 3D") -> Any:
    """3D-график треков (сплошные линии) и эталона (пунктир). Сохраняет PNG, если задан ``path``."""
    plt = _matplotlib()
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    for track in tracks:
        pts = sorted(track.points, key=lambda p: p.timestamp)
        X = np.array([p.position for p in pts], dtype=float)
        ax.plot(X[:, 0], X[:, 1], X[:, 2], color="tab:blue",
                label=f"трек {track.track_id}")
        ax.scatter(X[0, 0], X[0, 1], X[0, 2], color="tab:blue", marker="o", s=20)
    if truth_points:
        for k, points in enumerate(truth_points):
            X = np.asarray(points, dtype=float).reshape(-1, 3)
            ax.plot(X[:, 0], X[:, 1], X[:, 2], color="tab:red", linestyle="--",
                    label=f"эталон {k + 1}")
    ax.set_xlabel("X, м")
    ax.set_ylabel("Y, м")
    ax.set_zlabel("Z, м")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    if path is not None:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig


def overlay_track_on_frame(frame: np.ndarray, cam: CameraCalibration,
                           positions: list[np.ndarray], color=(0, 255, 0)) -> np.ndarray:
    """Наложение: репроекция позиций трека на кадр камеры (копия кадра с отметками)."""
    import cv2
    out = frame.copy()
    for X in positions:
        try:
            u, v = project_point(cam, np.asarray(X, dtype=float))
        except ValueError:
            continue
        if 0 <= u < out.shape[1] and 0 <= v < out.shape[0]:
            cv2.circle(out, (int(round(u)), int(round(v))), 6, color, 2, cv2.LINE_AA)
            cv2.drawMarker(out, (int(round(u)), int(round(v))), color,
                           cv2.MARKER_CROSS, 12, 1)
    return out


def tracks_to_csv(tracks: list[Track], path: str | Path) -> None:
    """CSV: ``track_id, timestamp, x, y, z, vx, vy, vz, views, mean_reproj_px``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "timestamp", "x", "y", "z",
                    "vx", "vy", "vz", "views", "mean_reproj_px"])
        for track in tracks:
            for pt in sorted(track.points, key=lambda q: q.timestamp):
                pos = np.asarray(pt.position, dtype=float)
                vel = np.asarray(pt.velocity, dtype=float)
                w.writerow([track.track_id, f"{pt.timestamp:.6f}",
                            *[f"{v:.6f}" for v in pos], *[f"{v:.6f}" for v in vel],
                            "|".join(pt.views),
                            f"{pt.mean_reproj_px:.3f}" if np.isfinite(pt.mean_reproj_px) else ""])


def tracks_from_csv(path: str | Path) -> list[Track]:
    """Чтение треков из CSV, записанного ``tracks_to_csv``."""
    from .types import TrackPoint
    tracks: dict[int, Track] = {}
    with Path(path).open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            tid = int(row["track_id"])
            track = tracks.setdefault(tid, Track(track_id=tid, points=[]))
            reproj = row.get("mean_reproj_px", "")
            track.points.append(TrackPoint(
                timestamp=float(row["timestamp"]),
                position=np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
                velocity=np.array([float(row["vx"]), float(row["vy"]), float(row["vz"])]),
                views=(row.get("views") or "").split("|") if row.get("views") else [],
                mean_reproj_px=float(reproj) if reproj not in ("", None) else float("nan"),
            ))
    return [tracks[k] for k in sorted(tracks)]


def truth_to_csv(truth_points: list[list[np.ndarray]], labels: list[str],
                 frame_times: list[float], path: str | Path) -> None:
    """CSV эталона: ``label, timestamp, x, y, z`` (единицы калибровки)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "timestamp", "x", "y", "z"])
        for label, points in zip(labels, truth_points):
            arr = np.asarray(points, dtype=float).reshape(-1, 3)
            for i, pos in enumerate(arr):
                t = frame_times[i] if i < len(frame_times) else float(i)
                w.writerow([label, f"{t:.6f}", f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}"])


def truth_from_csv(path: str | Path) -> tuple[dict[str, np.ndarray], list[float]]:
    """Чтение эталона: возвращает (``{label: (N,3)}``, отсортированные таймкоды)."""
    rows: dict[str, list[tuple[float, np.ndarray]]] = {}
    times: set[float] = set()
    with Path(path).open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = float(row["timestamp"])
            rows.setdefault(row["label"], []).append(
                (t, np.array([float(row["x"]), float(row["y"]), float(row["z"])]))
            )
            times.add(t)
    grid = sorted(times)
    index = {t: i for i, t in enumerate(grid)}
    truth = {}
    for label, items in rows.items():
        arr = np.full((len(grid), 3), np.nan)
        for t, pos in items:
            arr[index[t]] = pos
        truth[label] = arr
    return truth, grid


def tracks_to_json(tracks: list[Track], path: str | Path,
                   meta: dict[str, Any] | None = None) -> None:
    """JSON: ``{"meta": ..., "tracks": [Track.to_dict()]}``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta or {}, "tracks": [t.to_dict() for t in tracks]}
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
