#!/usr/bin/env python3
"""Interactive 3-camera + 3D trajectory viewer for the prepared ETH Dataset 3 run.

Expected layout:
  ~/PecStorm-mvus-reference/dataset3_3cam/result_dataset3_3cam.pkl
  ~/PecStorm-mvus-reference/dataset3_3cam/detections/cam{0,4,5}_mvus.txt
  ~/PecStorm-drone-videos/source/cam{0,4,5}/cam{0,4,5}.mp4

The MVUS time mapping is used to map each trajectory time to each camera's
original annotation frame: global_time = alpha[i] * camera_frame + beta[i].
Annotation frame numbers are one-based, so OpenCV frame index = frame - 1.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


HOME = Path.home()
PROJECT = HOME / "PecStorm-mvus-reference"
RESULT_PATH = PROJECT / "dataset3_3cam" / "result_dataset3_3cam.pkl"

CAMERAS = [
    {
        "name": "CAM 0 · GoPro 3",
        "video": HOME / "PecStorm-drone-videos/source/cam0/cam0.mp4",
        "detections": PROJECT / "dataset3_3cam/detections/cam0_mvus.txt",
    },
    {
        "name": "CAM 4 · Sony 5100",
        "video": HOME / "PecStorm-drone-videos/source/cam4/cam4.mp4",
        "detections": PROJECT / "dataset3_3cam/detections/cam4_mvus.txt",
    },
    {
        "name": "CAM 5 · Sony G",
        "video": HOME / "PecStorm-drone-videos/source/cam5/cam5.mp4",
        "detections": PROJECT / "dataset3_3cam/detections/cam5_mvus.txt",
    },
]


def load_detections(path: Path) -> dict[int, tuple[float, float]]:
    """Read MVUS detection text file: x y frame_id."""
    if not path.is_file():
        raise FileNotFoundError(f"Detection file not found: {path}")

    raw = np.loadtxt(path, dtype=float)
    raw = np.atleast_2d(raw)
    if raw.shape[1] < 3:
        raise ValueError(f"Expected at least 3 columns (x y frame), got {raw.shape} in {path}")

    result: dict[int, tuple[float, float]] = {}
    for row in raw:
        x, y, frame = float(row[0]), float(row[1]), float(row[2])
        if np.isfinite([x, y, frame]).all():
            result[int(round(frame))] = (x, y)
    return result


def read_video_frame(cap: cv2.VideoCapture, frame_index: int):
    """Seek to a zero-based source-video frame and return RGB image or None."""
    if frame_index < 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, bgr = cap.read()
    if not ok or bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    if not RESULT_PATH.is_file():
        raise FileNotFoundError(
            f"MVUS result not found: {RESULT_PATH}\n"
            "Run `python run_dataset3_compat.py` first."
        )

    # The pickle refers to reconstruction.common.Scene; make MVUS importable.
    sys.path.insert(0, str(PROJECT))
    sys.path.insert(0, str(PROJECT / "multiviewunsynch"))

    # NumPy 2.x removed np.asfarray; keep unpickling legacy MVUS objects compatible.
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

    with RESULT_PATH.open("rb") as file:
        flight = pickle.load(file)

    xyz = np.asarray(flight.out["reconst_tran"], dtype=float)
    trajectory_frames = np.asarray(flight.traj[0], dtype=float)
    alpha = np.asarray(flight.alpha, dtype=float)
    beta = np.asarray(flight.beta, dtype=float)

    if xyz.ndim != 2 or xyz.shape[0] != 3:
        raise ValueError(f"Expected reconst_tran shape (3, N), got {xyz.shape}")
    if xyz.shape[1] != trajectory_frames.size:
        raise ValueError("Trajectory coordinates and trajectory frame array have different lengths.")
    if alpha.size != 3 or beta.size != 3 or np.any(alpha == 0):
        raise ValueError("Expected three valid MVUS alpha/beta synchronization parameters.")

    valid = np.isfinite(trajectory_frames) & np.isfinite(xyz).all(axis=0)
    trajectory_frames = trajectory_frames[valid]
    xyz = xyz[:, valid]
    if trajectory_frames.size < 2:
        raise ValueError("Not enough valid 3D trajectory points to display.")

    detections = []
    captures = []
    for camera in CAMERAS:
        if not camera["video"].is_file():
            raise FileNotFoundError(f"Video not found: {camera['video']}")
        detections.append(load_detections(camera["detections"]))
        cap = cv2.VideoCapture(str(camera["video"]))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {camera['video']}")
        captures.append(cap)

    lengths = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures]
    fps_values = [float(cap.get(cv2.CAP_PROP_FPS)) for cap in captures]
    print("Loaded MVUS reconstruction:", xyz.shape[1], "3D points")
    for camera, count, fps in zip(CAMERAS, lengths, fps_values):
        print(f"{camera['name']}: {count} video frames, {fps:.3f} FPS")
    print("Playback uses MVUS-refined alpha/beta synchronization.")
    print("Coordinates are in the MVUS reconstruction coordinate system, not yet RTK-aligned metres.")

    # Dark desktop dashboard.
    fig = plt.figure(figsize=(16, 10), facecolor="#111827")
    fig.suptitle("PECSTORM  |  MULTI-CAMERA DRONE TRACKING", color="white", fontsize=16, fontweight="bold")
    grid = fig.add_gridspec(
        2, 3,
        left=0.035, right=0.975, top=0.90, bottom=0.17,
        height_ratios=[1.0, 1.18], hspace=0.22, wspace=0.08,
    )

    video_axes = []
    image_artists = []
    detection_artists = []
    info_artists = []
    for i, camera in enumerate(CAMERAS):
        ax = fig.add_subplot(grid[0, i])
        ax.set_facecolor("#0b1220")
        ax.set_title(camera["name"], color="white", fontsize=11, pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
        initial = read_video_frame(captures[i], 0)
        if initial is None:
            initial = np.zeros((540, 960, 3), dtype=np.uint8)
        image = ax.imshow(initial)
        marker = ax.scatter([], [], s=180, facecolors="none", edgecolors="#ff3b30", linewidths=2.5)
        cross = ax.scatter([], [], s=25, c="#ff3b30", marker="+")
        label = ax.text(
            0.02, 0.97, "", transform=ax.transAxes, va="top", ha="left",
            color="white", fontsize=9,
            bbox={"facecolor": "#111827", "alpha": 0.78, "edgecolor": "none", "pad": 4},
        )
        video_axes.append(ax)
        image_artists.append(image)
        detection_artists.append((marker, cross))
        info_artists.append(label)

    ax3d = fig.add_subplot(grid[1, :], projection="3d")
    ax3d.set_facecolor("#111827")
    ax3d.tick_params(colors="#d1d5db", labelsize=8)
    ax3d.xaxis.label.set_color("#e5e7eb")
    ax3d.yaxis.label.set_color("#e5e7eb")
    ax3d.zaxis.label.set_color("#e5e7eb")
    ax3d.set_title("3D RECONSTRUCTED TRAJECTORY", color="white", fontsize=11, pad=12)
    ax3d.set_xlabel("X (MVUS units)")
    ax3d.set_ylabel("Y (MVUS units)")
    ax3d.set_zlabel("Z (MVUS units)")
    ax3d.grid(True, alpha=0.25)

    x, y, z = xyz
    ax3d.plot(x, y, z, color="#64748b", linewidth=1.5, alpha=0.8)
    trail, = ax3d.plot([], [], [], color="#38bdf8", linewidth=2.5)
    current_point = ax3d.scatter([], [], [], color="#facc15", s=65, depthshade=False)
    ax3d.scatter([x[0]], [y[0]], [z[0]], color="#22c55e", s=35, label="Start")
    ax3d.scatter([x[-1]], [y[-1]], [z[-1]], color="#ef4444", s=35, label="End")
    ax3d.legend(loc="upper left", fontsize=8)
    ranges = np.array([np.ptp(x), np.ptp(y), np.ptp(z)], dtype=float)
    ranges[ranges == 0] = 1.0
    ax3d.set_box_aspect(ranges)

    # Coordinate/status strip.
    status = fig.text(
        0.50, 0.115, "", ha="center", va="center", color="#e5e7eb", fontsize=10,
        bbox={"facecolor": "#1f2937", "edgecolor": "#374151", "boxstyle": "round,pad=0.5"},
    )

    slider_ax = fig.add_axes([0.18, 0.055, 0.69, 0.035], facecolor="#1f2937")
    frame_slider = Slider(
        slider_ax, "Frame", 0, trajectory_frames.size - 1,
        valinit=0, valstep=1, color="#38bdf8",
    )
    frame_slider.label.set_color("white")
    frame_slider.valtext.set_color("white")

    button_ax = fig.add_axes([0.045, 0.045, 0.105, 0.055])
    play_button = Button(button_ax, "▶  Play", color="#1f2937", hovercolor="#374151")
    play_button.label.set_color("white")

    reset_ax = fig.add_axes([0.885, 0.045, 0.085, 0.055])
    reset_button = Button(reset_ax, "Reset", color="#1f2937", hovercolor="#374151")
    reset_button.label.set_color("white")

    state = {"playing": False, "updating_slider": False}

    def update(index: int) -> None:
        index = int(np.clip(index, 0, trajectory_frames.size - 1))
        global_frame = float(trajectory_frames[index])

        for i, camera in enumerate(CAMERAS):
            # MVUS maps a camera frame to the reference/global timeline:
            # global = alpha[i] * camera_frame + beta[i].
            camera_frame = int(round((global_frame - beta[i]) / alpha[i]))
            video_index = camera_frame - 1  # Dataset annotation frame IDs are 1-based.
            rgb = read_video_frame(captures[i], video_index)

            if rgb is None:
                image_artists[i].set_data(np.zeros((540, 960, 3), dtype=np.uint8))
                info_artists[i].set_text(f"frame {camera_frame} | VIDEO FRAME UNAVAILABLE")
                marker, cross = detection_artists[i]
                marker.set_offsets(np.empty((0, 2)))
                cross.set_offsets(np.empty((0, 2)))
                continue

            image_artists[i].set_data(rgb)
            point = detections[i].get(camera_frame)
            marker, cross = detection_artists[i]
            if point is not None:
                px, py = point
                marker.set_offsets([[px, py]])
                cross.set_offsets([[px, py]])
                detection_text = f"2D detection: x={px:.1f}, y={py:.1f}"
            else:
                marker.set_offsets(np.empty((0, 2)))
                cross.set_offsets(np.empty((0, 2)))
                detection_text = "NO 2D DETECTION IN THIS FRAME"
            info_artists[i].set_text(f"video frame {camera_frame}\n{detection_text}")

        trail.set_data(x[: index + 1], y[: index + 1])
        trail.set_3d_properties(z[: index + 1])
        current_point._offsets3d = ([x[index]], [y[index]], [z[index]])

        status.set_text(
            f"Reference frame: {global_frame:.1f}     |     "
            f"X: {x[index]:.4f}     Y: {y[index]:.4f}     Z: {z[index]:.4f}     |     "
            f"Point {index + 1}/{trajectory_frames.size}"
        )

        fig.canvas.draw_idle()

    def on_slider(value):
        if not state["updating_slider"]:
            update(int(round(value)))

    def step_playback():
        if not state["playing"]:
            return
        current = int(round(frame_slider.val))
        if current >= trajectory_frames.size - 1:
            state["playing"] = False
            play_button.label.set_text("▶  Play")
            return
        next_index = current + 1
        state["updating_slider"] = True
        frame_slider.set_val(next_index)
        state["updating_slider"] = False
        update(next_index)

    timer = fig.canvas.new_timer(interval=17)  # Approximately 60 reference frames/second.
    timer.add_callback(step_playback)

    def toggle_play(_event):
        state["playing"] = not state["playing"]
        if state["playing"]:
            if int(round(frame_slider.val)) >= trajectory_frames.size - 1:
                state["updating_slider"] = True
                frame_slider.set_val(0)
                state["updating_slider"] = False
                update(0)
            play_button.label.set_text("Ⅱ  Pause")
            timer.start()
        else:
            play_button.label.set_text("▶  Play")
            timer.stop()
        fig.canvas.draw_idle()

    def reset(_event):
        state["playing"] = False
        timer.stop()
        play_button.label.set_text("▶  Play")
        state["updating_slider"] = True
        frame_slider.set_val(0)
        state["updating_slider"] = False
        update(0)

    frame_slider.on_changed(on_slider)
    play_button.on_clicked(toggle_play)
    reset_button.on_clicked(reset)

    update(0)
    fig.canvas.manager.set_window_title("PecStorm — Drone Tracking")

    try:
        plt.show()
    finally:
        timer.stop()
        for cap in captures:
            cap.release()


if __name__ == "__main__":
    main()
