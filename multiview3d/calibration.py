"""Калибрация камер: внутренние параметры и внешние параметры в общей системе координат.

Используется открытая модель камеры OpenCV:

    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],  dist = [k1, k2, p1, p2, k3]

Внешние параметрыcamera = (R, t): X_cam = R @ X_world + t.

Практический приём для нескольких камер: все ракурсы калибровочного
(шахматного или кругового) полигонаSolution решаются в одной и той же системе
координат полигона (её центр - начало координат, плоскость полигона z = 0).
Тогда внешние параметры всех камер автоматически согласованы между собой.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .camera import (
    baseline_length,
    epipolar_distance,
    epipolar_distance_matrix,
    project_points,
    triangulate_rays,
)
from .types import CameraCalibration


def rot_from_vector(v: np.ndarray) -> np.ndarray:
    """Матрица поворота из вектора вращения OpenCV (3,)."""
    R, _ = cv2.Rodrigues(np.asarray(v, dtype=np.float64).reshape(3, 1))
    return R


def rot_to_vector(R: np.ndarray) -> np.ndarray:
    """Вектор вращения OpenCV (3,) из матрицы поворота."""
    v, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64).reshape(3, 3))
    return v.reshape(3)


def _rot_translation_to_vector(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Параметры внешних параметров [rvec | t] (6,) для bандл-аджастмента."""
    return np.concatenate([rot_to_vector(R), np.asarray(t, dtype=np.float64).reshape(3)])


def board_object_points(
    pattern: tuple[int, int], square_size_mm: float, kind: str = "chessboard"
) -> np.ndarray:
    """3D-точки внутренних углов полигона в его собственной системе координат.

    pattern - число внутренних углов по (ширине, высоте).
    """
    cols, rows = pattern
    obj = np.zeros((rows * cols, 3), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            obj[i * cols + j] = (j * square_size_mm, i * square_size_mm, 0.0)
    if kind == "chessboard":
        obj -= ((cols - 1) * square_size_mm / 2.0, (rows - 1) * square_size_mm / 2.0, 0.0)
    return obj


def find_board(
    image_bgr: np.ndarray,
    pattern: tuple[int, int],
    kind: str = "chessboard",
) -> np.ndarray | None:
    """Находит углы калибровочного полигона; возвращает None, если не найден."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    flags = 0
    if kind == "chessboard":
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    elif kind == "circles":
        ok, corners = cv2.findCirclesGrid(gray, pattern, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    else:
        raise ValueError(f"неизвестный тип полигона: {kind}")
    if not ok:
        return None
    corners = cv2.cornerSubPix(
        gray, corners, (5, 5), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
    )
    return corners.reshape(-1, 2).astype(np.float64)


def collect_views(
    image_paths: Iterable[Path | str] | Sequence[np.ndarray],
    pattern: tuple[int, int],
    kind: str = "chessboard",
    min_board_pixels: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Собирает ракурсы: (объектные точки, пиксельные точки, имена кадров)."""
    paths = list(image_paths)
    is_paths = isinstance(paths[0], (str, Path))
    views_img: list[np.ndarray] = []
    views_obj: list[np.ndarray] = []
    used: list[str] = []
    skipped: list[str] = []
    for item in paths:
        if is_paths:
            img = cv2.imread(str(item), cv2.IMREAD_COLOR)
            if img is None:
                skipped.append(str(item))
                continue
            name = str(item)
        else:
            img = item
            name = f"frame#{len(used)}"
        corners = find_board(img, pattern, kind)
        if corners is None:
            skipped.append(name)
            continue
        if min_board_pixels > 0:
            span = corners.max(axis=0) - corners.min(axis=0)
            if span.min() < min_board_pixels:
                skipped.append(name)
                continue
        views_obj.append(board_object_points(pattern, 1.0, kind))
        views_img.append(corners)
        used.append(name)
    if not views_img:
        raise ValueError(
            "полигон не найден ни на одном кадре; пропущено: " + ", ".join(skipped[:5])
        )
    return views_obj, views_img, used


def calibrate_intrinsics(
    image_sizes: Sequence[tuple[int, int]],
    views_obj: list[np.ndarray],
    views_img: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Внутренние параметры по набору ракурсов. Возвращает (K, dist, RMS)."""
    # OpenCV 5 требует vector<vector<Point3f>> / vector<vector<Point2f>>;
    # RMS в OpenCV 5 не возвращается — считаем сами по rvecs/tvecs.
    obj_f = [np.asarray(o, np.float32).reshape(-1, 3) for o in views_obj]
    img_f = [np.asarray(c, np.float32).reshape(-1, 2) for c in views_img]
    ok, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_f, img_f, tuple(image_sizes[0]), None, None
    )
    if not ok:
        raise ValueError("cv2.calibrateCamera вернул ошибку")
    sq = 0.0
    n = 0
    for o, c, rv, tv in zip(obj_f, img_f, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
        sq += float(((proj.reshape(-1, 2) - c) ** 2).sum(axis=1).sum())
        n += len(c)
    rms = (sq / max(n, 1)) ** 0.5
    return K.astype(np.float64), dist.astype(np.float64).reshape(-1), float(rms)


def calibration_from_pose(
    name: str,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    width: int,
    height: int,
    unit: float = 1.0,
) -> CameraCalibration:
    """CameraCalibration из векторов rotation/translation OpenCV.

    ``unit`` — множитель перевода ``tvec`` в метры (1.0 — метры, 0.001 — мм).
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return CameraCalibration(
        name=name,
        K=np.asarray(K, dtype=np.float64),
        dist=np.asarray(dist, dtype=np.float64).reshape(-1),
        R=R,
        t=np.asarray(tvec, dtype=np.float64).reshape(3),
        width=int(width),
        height=int(height),
        unit=unit,
    )


def calibrate_camera_from_images(
    name: str,
    image_paths: Sequence[Path | str] | Sequence[np.ndarray],
    pattern: tuple[int, int],
    square_size_mm: float = 30.0,
    kind: str = "chessboard",
    unit: float = 1.0,
) -> tuple[CameraCalibration, dict[str, Any]]:
    """Калибрация одной камеры по изображениям полигона (в системе координат полигона).

    Принимает пути к файлам либо готовые кадры (BGR-массивы).
    ``square_size_mm`` — сторона клетки полигона в миллиметрах; мир при этом
    переводится в метры (``t`` в метрах, ``unit == 1.0``).
    """
    views_obj, views_img, used = collect_views(image_paths, pattern, kind)
    size = _image_size(image_paths[0])
    obj = [o * (square_size_mm * 0.001) for o in views_obj]
    K, dist, rms = calibrate_intrinsics([size], obj, views_img)
    ok, rvec, tvec = cv2.solvePnP(obj[0], views_img[0], K, dist)
    if not ok:
        raise ValueError("cv2.solvePnP не сошёлся")
    cam = calibration_from_pose(name, K, dist, rvec, tvec, size[0], size[1], unit)
    return cam, {
        "reprojection_rms_px": rms,
        "views_used": len(used),
        "views_skipped": len(image_paths) - len(used),
        "images": used,
    }


def _image_size(path: Path | str | np.ndarray) -> tuple[int, int]:
    if isinstance(path, np.ndarray):
        h, w = path.shape[:2]
        return (w, h)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"не удалось прочитать изображение: {path}")
    h, w = img.shape[:2]
    return (w, h)


def solve_extrinsics_from_points(
    camera: CameraCalibration,
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    refine: bool = True,
) -> CameraCalibration:
    """Внешние параметры камеры по 3D-точкам и их пиксельным проекциям.

    points_3d задаются в общей (мировой) системе координат, поэтому решённые
    R, t сразу согласованы с другими камерами.
    """
    points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    points_2d = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    if points_3d.shape != points_2d.shape[:1] + (3,) or len(points_3d) < 4:
        raise ValueError("нужно не менее 4 соответствий 3D-точка <-> пиксель")
    ok, rvec, tvec = cv2.solvePnP(
        points_3d, points_2d, camera.K, camera.dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise ValueError("cv2.solvePnP не сошёлся")
    R, _ = cv2.Rodrigues(rvec)
    return CameraCalibration(
        name=camera.name,
        K=camera.K.copy(),
        dist=camera.dist.copy(),
        R=R,
        t=tvec.reshape(3),
        width=camera.width,
        height=camera.height,
        unit=camera.unit,
    )


def reprojection_rms(camera: CameraCalibration, points_3d, points_2d) -> float:
    """Среднеквадратичная ошибка репроекции в пикселях."""
    projected = project_points(camera, np.asarray(points_3d, dtype=np.float64))
    diff = projected - np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.einsum("ij,ij->i", diff, diff))))


def board_full_points(
    board_size: tuple[int, int],
    square_size: float,
    marker_size: int = 0,
) -> np.ndarray:
    """Объектные точки шахматного поля с дополнительными угловыми маркерами.

    Маркеры — точки вне прямоугольника поля (по 4 углам), они дают более
    устойчивую оценку ориентации камеры. Координаты маркеров задаются в тех же
    единицах, что и ``square_size``.
    """
    points = board_object_points(board_size, square_size)
    if marker_size <= 0:
        return points
    cols, rows = board_size
    w = cols * square_size
    h = rows * square_size
    m = marker_size
    extra = [
        (-m, -m, 0.0),
        (w + m, -m, 0.0),
        (-m, h + m, 0.0),
        (w + m, h + m, 0.0),
    ]
    return np.vstack([points, np.asarray(extra, dtype=np.float64)])


def bundle_adjust(
    cameras: list[CameraCalibration],
    points_3d: np.ndarray,
    observations: list[tuple[int, int]],
    observed_pixels: np.ndarray,
    *,
    adjust_intrinsics: bool = False,
) -> dict:
    """Бандл-аджастментмент: уточняет экстринсики (и опционально интринсики) и 3D-точки.

    :param observations: список ``(index_camera, index_point)`` для каждого наблюдения.
    :param observed_pixels: массив ``(N, 2)`` в том же порядке, что и ``observations``.
    :return: словарь с обновлёнными камерами, точками и среднеквадратичной ошибкой.
    """
    from scipy.optimize import least_squares

    observations = list(observations)
    pts = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3).copy()
    pixels = np.asarray(observed_pixels, dtype=np.float64).reshape(-1, 2)
    cams = [copy.deepcopy(c) for c in cameras]

    n_cam = len(cams)
    n_point = pts.shape[0]
    n_ext = 6 * n_cam
    int_per_cam = 4 if adjust_intrinsics else 0
    n_int = int_per_cam * n_cam

    ext0 = np.concatenate([_rot_translation_to_vector(c.R, c.t) for c in cams])
    int0 = (
        np.concatenate(
            [
                np.array(
                    [
                        c.K[0, 0],
                        c.K[1, 1],
                        0.0 if abs(c.K[0, 1]) < 1e-9 else c.K[0, 1] / 2.0,
                        (c.width - 1) / 2.0 if c.K[0, 2] <= 0 else c.K[0, 2],
                    ],
                    dtype=np.float64,
                )
                for c in cams
            ]
        )
        if adjust_intrinsics
        else np.empty(0, dtype=np.float64)
    )
    x0 = np.concatenate([ext0, int0, pts.ravel()])

    def build(x: np.ndarray) -> tuple[list[CameraCalibration], np.ndarray]:
        ext = x[:n_ext]
        off = n_ext
        out = []
        for i, cam in enumerate(cams):
            vec = ext[6 * i : 6 * i + 6]
            R = rot_from_vector(vec[:3])
            K = cam.K.copy()
            if adjust_intrinsics:
                kx, ky, skew, cx = x[off + int_per_cam * i : off + int_per_cam * i + int_per_cam]
                K = np.array([[kx, skew, cx], [0.0, ky, (cam.height - 1) / 2.0], [0.0, 0.0, 1.0]])
            out.append(
                CameraCalibration(
                    name=cam.name,
                    K=K,
                    dist=cam.dist,
                    R=R,
                    t=vec[3:6],
                    width=cam.width,
                    height=cam.height,
                    unit=cam.unit,
                )
            )
        return out, x[n_ext + n_int:].reshape(-1, 3)

    cam_idx = np.array([c for c, _ in observations], dtype=np.int64)
    pt_idx = np.array([p for _, p in observations], dtype=np.int64)

    def residuals(x: np.ndarray) -> np.ndarray:
        cams_i, pts_i = build(x)
        projected = np.empty((len(observations), 2), dtype=np.float64)
        for i, cam in enumerate(cams_i):
            mask = cam_idx == i
            if not np.any(mask):
                continue
            projected[mask] = project_points(cam, pts_i[pt_idx[mask]])
        return (projected - pixels).ravel()

    result = least_squares(residuals, x0, method="lm", max_nfev=200)
    new_cams, new_pts = build(result.x)
    rms = float(np.sqrt(np.mean(result.fun ** 2)))
    return {"cameras": new_cams, "points_3d": new_pts, "rms_px": rms, "success": bool(result.success)}


def correspondence_epipolar_errors(
    cam_a: CameraCalibration,
    cam_b: CameraCalibration,
    pixels_a: np.ndarray,
    pixels_b: np.ndarray,
) -> np.ndarray:
    """Матрица симметричных эпиполярных ошибок ``(N_a, N_b)`` в пикселях.

    Сумма расстояния от точки ``a`` до эпиполярной линии в ``b`` и расстояния от
    точки ``b`` до эпиполярной линии в ``a``. Малое значение — необходимое условие
    того, что обе детекции описывают одну и ту же точку пространства.
    """
    a = np.asarray(pixels_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(pixels_b, dtype=np.float64).reshape(-1, 2)
    forward = epipolar_distance_matrix(cam_a, cam_b, a, b)  # (N_a, N_b)
    backward = epipolar_distance_matrix(cam_b, cam_a, b, a)  # (N_b, N_a)
    return forward + backward.T


def quality_report(
    cameras: list[CameraCalibration],
    calibration_views: dict[int, int] | None = None,
    reproj_rms_px: dict[int, float] | None = None,
) -> dict:
    """Сводка качества калибрации: число видов, RMS, межкамевые базисы.

    Возвращает словарь, пригодный для печати и JSON-сериализации.
    """
    rep: dict = {"cameras": [], "baselines": {}, "warnings": []}
    for i, cam in enumerate(cameras):
        rep["cameras"].append(
            {
                "index": i,
                "name": cam.name,
                "views": int((calibration_views or {}).get(i, 0)),
                "reproj_rms_px": float((reproj_rms_px or {}).get(i, float("nan"))),
                "focal_px": float((cam.K[0, 0] + cam.K[1, 1]) / 2.0),
                "principal_point": [float(cam.K[0, 2]), float(cam.K[1, 2])],
                "distortion": [float(x) for x in cam.dist],
            }
        )
        if (calibration_views or {}).get(i, 0) < 3:
            rep["warnings"].append(f"камера {cam.name}: меньше 3 калибровочных видов")
        if abs(cam.K[0, 0]) < 1.0:
            rep["warnings"].append(f"камера {cam.name}: фокус не определён")

    for i in range(len(cameras)):
        for j in range(i + 1, len(cameras)):
            b = baseline_length(cameras[i], cameras[j])
            rep["baselines"][f"{cameras[i].name}→{cameras[j].name}"] = round(b, 4)
            if b < 1e-3:
                rep["warnings"].append(
                    f"базис {cameras[i].name}→{cameras[j].name} ≈ 0: триангуляция неустойчива"
                )
    return rep


def save_calibration(path: str | Path, cameras: list[CameraCalibration], meta: dict | None = None) -> None:
    """Сохраняет калибрацию всех камер в JSON-файл."""
    payload = {
        "version": 1,
        "meta": meta or {},
        "cameras": [cam.to_dict() for cam in cameras],
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_calibration(path: str | Path) -> tuple[list[CameraCalibration], dict]:
    """Читает калибрацию из JSON-файла, возвращает ``(cameras, meta)``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cameras = [CameraCalibration.from_dict(c) for c in payload["cameras"]]
    return cameras, dict(payload.get("meta", {}))
