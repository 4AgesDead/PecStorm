"""Слияние CSV детекций нескольких моделей (ансамбль по боксам).

Правило: боксы всех моделей объединяются, дубликаты (IoU ≥ порога) схлопываются,
остаётся бокс с большим score. Полезно, когда разные модели находят разные
кадры: объединение поднимает recall, а `single_target` дальше берёт лучший бокс.

Запуск:
    python tools/merge_detections.py --inputs out/truck/det1.csv,out/truck/det2.csv `
        --out out/truck/detections_ens.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FIELDS = ["camera", "frame", "timestamp", "score", "label",
          "x1", "y1", "x2", "y2", "cx", "cy"]


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def load(path: Path) -> dict[tuple[str, int], list[dict]]:
    out: dict[tuple[str, int], list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("x1") in (None, ""):
                continue
            out[(r["camera"], int(r["frame"]))].append({
                "row": r,
                "box": (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])),
                "score": float(r["score"]),
            })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Слияние детекций нескольких моделей")
    ap.add_argument("--inputs", required=True, help="CSV через запятую")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou", type=float, default=0.5, help="порог дубликата")
    args = ap.parse_args(argv)

    paths = [Path(p.strip()) for p in args.inputs.split(",") if p.strip()]
    for p in paths:
        if not p.exists():
            print(f"нет файла {p}", file=sys.stderr)
            return 2
    merged: dict[tuple[str, int], list[dict]] = defaultdict(list)
    total = 0
    for p in paths:
        for key, boxes in load(p).items():
            merged[key].extend(boxes)
            total += len(boxes)

    rows = []
    for key, boxes in merged.items():
        boxes.sort(key=lambda d: -d["score"])
        kept: list[dict] = []
        for b in boxes:
            if all(iou(b["box"], k["box"]) < args.iou for k in kept):
                kept.append(b)
        for k in kept:
            rows.append((key[1], key[0], k["row"]))
    rows.sort(key=lambda t: (t[0], t[1]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for _, _, row in rows:
            w.writerow({k: row.get(k, "") for k in FIELDS})
    print(f"вход: {total} боксов из {len(paths)} файлов -> "
          f"{len(rows)} после слияния (IoU≥{args.iou})")
    print(f"записано: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
