"""Импорт CSV экспорта Unreal Engine (данные «Пеленга», Space Hackathon 2.0).

Формат: CP1251, разделитель ';'. Камеры:
``Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;<focal_mm>``
со строкой метаданных ``матрица;<sw>*<sh>мм``. Цель (GT): ``Frame;Loc_X;Loc_Y;Loc_Z``.

Оригиналы файлов не изменяются. Повреждённые Excel-ячейки («авг.13», «июл.25»)
собираются в ``invalid_cells``; восстановление — sidecar ``ground_truth_repairs.json``
с флагом ``verified_by_organizer``.

Конверсия (рабочая гипотеза, см. docs/OPEN_QUESTIONS.md):
* единицы UE — сантиметры -> метры (x0.01);
* правая система: ``X_ue=X_u, Y_ue=-Y_u, Z_ue=Z_u`` (детерминант +1);
* локальные оси камеры UE: X вперёд, Y вправо, Z вверх; OpenCV: X вправо,
  Y вниз, Z вперёд; смена базиса ``M_s`` (det=+1);
* ``R_ue`` по формуле FRotator из UE4 (roll->X, pitch->Y, yaw->Z);
* OpenCV: ``R_w2c = M_s @ R_ue.T``, ``t = -R_w2c @ C_ue`` в метрах;
* intrinsics: ``fx = focal_mm * width / sensor_w``, ``fy = focal_mm * height / sensor_h``,
  principal point в центре, дисторсия нулевая (не подтверждена организаторами).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .types import CameraCalibration

CM_TO_M = 0.01


@dataclass
class UnrealRow:
    frame: int
    loc: tuple[float, float, float]      # (X, Y, Z) в сантиметрах UE
    rot: tuple[float, float, float]      # (roll, pitch, yaw) в градусах UE


@dataclass
class ParsedCsv:
    rows: list[UnrealRow]
    focal_mm: float | None = None
    sensor_mm: tuple[float, float] | None = None
    invalid_cells: list[dict] = field(default_factory=list)
    partial: dict[int, dict[str, float]] = field(default_factory=dict)
    source_path: str = ""


def _read_lines(path: str | Path) -> list[str]:
    p = Path(path)
    data = p.read_bytes()
    text = data.decode("cp1251", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _num(cell: str, ctx: dict, invalid: list[dict]) -> float | None:
    cell = cell.strip()
    if not cell:
        return None
    try:
        return float(cell)
    except ValueError:
        invalid.append({**ctx, "raw_value": cell})
        return None


_SENSOR_RE = re.compile(r"([\d.,]+)\*([\d.,]+)мм")


def parse_camera_csv(path: str | Path) -> ParsedCsv:
    """Читает CSV камеры. Металюки/битые ячейки -> ``invalid_cells``, без coercion в NaN."""
    lines = _read_lines(path)
    invalid: list[dict] = []
    focal_mm: float | None = None
    sensor_mm: tuple[float, float] | None = None
    rows: list[UnrealRow] = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        cells = line.split(";")
        first = cells[0].strip()
        if first == "Frame":
            try:
                focal_mm = float(cells[-1].strip())
            except (ValueError, IndexError):
                focal_mm = None
            continue
        m = _SENSOR_RE.search(line)
        if m:
            sensor_mm = (float(m.group(1).replace(",", ".")),
                         float(m.group(2).replace(",", ".")))
        if first == "матрица" or not first.isdigit():
            continue
        frame = int(first)
        vals: list[float | None] = []
        for j, cell in enumerate(cells[1:7]):
            vals.append(_num(cell, {"line": i + 1, "frame": frame, "column": j}, invalid))
        if any(v is None for v in vals):
            continue
        rows.append(UnrealRow(frame, (vals[0], vals[1], vals[2]), (vals[3], vals[4], vals[5])))
    return ParsedCsv(rows=rows, focal_mm=focal_mm, sensor_mm=sensor_mm,
                     invalid_cells=invalid, source_path=str(path))


def parse_track_csv(path: str | Path) -> ParsedCsv:
    """Читает GT-CSV цели. Битые ячейки обнаруживаются, но не подменяются молча."""
    lines = _read_lines(path)
    invalid: list[dict] = []
    rows: list[UnrealRow] = []
    partial: dict[int, dict[str, float]] = {}
    names = {0: "Loc_X", 1: "Loc_Y", 2: "Loc_Z"}
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        cells = line.split(";")
        first = cells[0].strip()
        if first == "Frame" or not first.isdigit():
            continue
        frame = int(first)
        vals = [_num(cells[j + 1] if j + 1 < len(cells) else "",
                     {"line": i + 1, "frame": frame, "column": names[j]}, invalid)
                for j in range(3)]
        if any(v is None for v in vals):
            partial[frame] = {names[j]: v for j, v in enumerate(vals) if v is not None}
            continue
        rows.append(UnrealRow(frame, (vals[0], vals[1], vals[2]), (0.0, 0.0, 0.0)))
    return ParsedCsv(rows=rows, invalid_cells=invalid, partial=partial, source_path=str(path))


def ue_rot_to_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float,
                     legacy: bool = False) -> np.ndarray:
    """Матрица осей UE (столбцы: fwd/right/up).

    ``legacy=True`` — историческая формула с зеркальным ``up`` (используется
    пайплайном scene1: его поправки `anchor_offset_views` подгонялись под неё).
    ``legacy=False`` — ортонормированная матрица (det=+1); проверено на scene2:
    даёт 5–7 px на уже измеренных кадрах и делает пригодной наклонную
    аэрокамеру cam2 (pitch ≈ −45°).
    """
    sp, cp = np.sin(np.radians(pitch_deg)), np.cos(np.radians(pitch_deg))
    sy, cy = np.sin(np.radians(yaw_deg)), np.cos(np.radians(yaw_deg))
    sr, cr = np.sin(np.radians(roll_deg)), np.cos(np.radians(roll_deg))
    fwd = np.array([cp * cy, cp * sy, sp])
    right = np.array([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp])
    if legacy:  # историческая (неортонормальная) формула
        up = np.array([cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp])
    else:
        up = np.array([-cr * sp * cy - sr * sy, -cr * sp * sy + sr * cy, cr * cp])
    return np.vstack((fwd, right, up)).T  # columns: fwd, right, up (UE world basis)


# Flip of the UE left-handed world into a right-handed frame (det = -1).
S_FLIP = np.diag([1.0, -1.0, 1.0])


def unreal_to_right(x_u: float, y_u: float, z_u: float) -> np.ndarray:
    """UE world point (cm) -> right-handed world point (meters, y mirrored)."""
    return np.array([x_u, -y_u, z_u]) * CM_TO_M


def build_extrinsics(row: UnrealRow, legacy: bool = False
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R_w2c, t, C): OpenCV extrinsics из строки UE-CSV, метры.

    ``legacy=False`` (по умолчанию, scene2): корректная ортонормальная матрица
    поворота (det=+1). ``legacy=True`` (scene1): историческая формула, где знак
    pitch инвертировался, а у ``up`` была зеркальна горизонтальная часть — две
    ошибки компенсировали друг друга при малых |pitch| (<10°), но ломали
    наклонные камеры. Пайплайн scene1 оставлен на legacy, потому что его
    поправки `anchor_offset_views`/`anchor_azimuth` подгонялись именно под неё.
    """
    roll, pitch, yaw = row.rot
    if legacy:
        axes = ue_rot_to_matrix(roll, -pitch, yaw, legacy=True)
    else:
        axes = ue_rot_to_matrix(roll, pitch, yaw)   # columns = fwd,right,up (UE)
    r_w2c = np.array([axes[:, 1], -axes[:, 2], axes[:, 0]]) @ S_FLIP
    c_w = unreal_to_right(*row.loc)
    t = -r_w2c @ c_w
    return r_w2c, t, c_w


def build_intrinsics(focal_mm: float, sensor_mm: tuple[float, float],
                     width: int, height: int) -> np.ndarray:
    """K по фокусу и сенсору UE (pinhole, principal point в центре).

    Исходные допущения: весь сенсор без crop/overscan, дисторсия нулевая.
    """
    fx = focal_mm * width / sensor_mm[0]
    fy = focal_mm * height / sensor_mm[1]
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fy, height / 2.0],
                     [0.0, 0.0, 1.0]])


def camera_from_row(row: UnrealRow, name: str, focal_mm: float,
                    sensor_mm: tuple[float, float], width: int, height: int,
                    legacy: bool = False) -> CameraCalibration:
    """``CameraCalibration`` (метрическая) для одной строки UE-CSV."""
    r_w2c, t, _ = build_extrinsics(row, legacy=legacy)
    return CameraCalibration(
        name=name,
        K=build_intrinsics(focal_mm, sensor_mm, width, height),
        dist=np.zeros(5),
        R=r_w2c,
        t=t,
        width=width,
        height=height,
        unit=1.0,
    )


@dataclass
class GroundTruth:
    frames: np.ndarray                 # (N,) int, номера кадров
    positions_m: np.ndarray            # (N,3) метры в правой системе UE_ue
    invalid_cells: list[dict]
    repaired_frames: list[int]
    source_path: str


def load_ground_truth(path: str | Path, repairs_path: str | Path | None = None,
                      strict: bool = False,
                      accept_unverified: bool | None = None) -> GroundTruth:
    """GT цели в метрах. Битые ячейки: strict — ошибка; иначе — sidecar override.

    ``repairs_path`` — JSON вида
    ``{"repairs": [{"frame": 402, "column": "Loc_X", "proposed_value": 8.13,
    "source_raw_value": "авг.13", "reason": "...", "verified_by_organizer": false}]}``.
    Override применяется только если ``verified_by_organizer`` явно true или передан
    ``accept_unverified`` (аргументом или полем ``accept_unverified: true`` в
    repairs-файле). Исходный CSV при этом не изменяется.
    """
    parsed = parse_track_csv(path)
    by_frame = {r.frame: r.loc for r in parsed.rows}
    repairs: dict[tuple[int, str], dict] = {}
    accept_from_file = False
    if repairs_path is not None and Path(repairs_path).exists():
        doc = json.loads(Path(repairs_path).read_text(encoding="utf-8"))
        accept_from_file = bool(doc.get("accept_unverified", False))
        for rec in doc.get("repairs", []):
            repairs[(int(rec["frame"]), rec["column"])] = rec
    if accept_unverified is None:
        accept_unverified = accept_from_file
    else:
        accept_unverified = bool(accept_unverified)
    invalid = list(parsed.invalid_cells)
    repaired: list[int] = []
    for cell in invalid:
        key = (cell["frame"], cell["column"])
        rec = repairs.get(key)
        if rec is None:
            if strict:
                raise ValueError(
                    f"GT повреждён: frame={cell['frame']} column={cell['column']} "
                    f"raw={cell['raw_value']!r}; repairs не заданы (strict=True)"
                )
            continue
        if not rec.get("verified_by_organizer", False) and not accept_unverified:
            if strict:
                raise ValueError(
                    f"override frame={cell['frame']} column={cell['column']} "
                    f"не подтверждён организаторами (verified_by_organizer=false)"
                )
            continue
        by_frame[key[0]] = _repair_value(rec, cell, by_frame, parsed.partial)
        repaired.append(key[0])
    frames = sorted(by_frame)
    pos = np.array([unreal_to_right(*by_frame[f]) for f in frames], dtype=float)
    return GroundTruth(np.array(frames, dtype=int), pos, invalid, repaired, str(path))


_MONTHS = {"янв": 1, "фев": 2, "мар": 3, "апр": 4, "май": 5, "июн": 6,
           "июл": 7, "авг": 8, "сент": 9, "окт": 10, "нояб": 11, "дек": 12}


def _repair_value(rec: dict, cell: dict, by_frame: dict, partial: dict) -> tuple[float, float, float]:
    """Восстановление строки GT: proposed_value заменяет только повреждённую колонку,
    остальные колонки берутся из успешно распарсенных частичных значений строки."""
    vals = dict(partial.get(rec["frame"], {}))
    if rec["frame"] in by_frame:
        x, y, z = by_frame[rec["frame"]]
        vals.setdefault("Loc_X", x)
        vals.setdefault("Loc_Y", y)
        vals.setdefault("Loc_Z", z)
    vals[rec["column"]] = float(rec["proposed_value"])
    return (vals["Loc_X"], vals["Loc_Y"], vals["Loc_Z"])


def propose_repairs(invalid_cells: list[dict]) -> list[dict]:
    """Гипотеза восстановления Excel date coercion: «авг.13» -> 8.13, «июл.25» -> 7.25.

    Месяц даёт целую часть, день — дробную (формат mm.dd). Гипотеза, требует
    подтверждения организаторами (``verified_by_organizer=false``).
    """
    out = []
    for cell in invalid_cells:
        raw = cell["raw_value"].lower().rstrip(".")
        m = re.match(r"^([а-яё]+)\.(\d+)$", raw)
        if not m:
            continue
        stem = m.group(1)
        day = int(m.group(2))
        month = next((v for k, v in _MONTHS.items() if stem.startswith(k)), None)
        if month is None:
            continue
        out.append({
            "frame": cell["frame"],
            "column": cell["column"],
            "source_raw_value": cell["raw_value"],
            "proposed_value": float(f"{month}.{day}"),
            "reason": (f"Excel date coercion: «{cell['raw_value']}» = "
                       f"{month}.{day} по формату mm.dd; гипотеза, не подтверждена"),
            "verified_by_organizer": False,
        })
    return out

