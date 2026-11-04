"""Мозаика 3x3 из 9 камер scene2 для визуальной проверки кадра."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

from multiview3d.scene2_dataset import Scene2Dataset


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Мозаика 9 камер scene2")
    ap.add_argument("--frames", default="520,900")
    ap.add_argument("--out-dir", default="out/scene2/mosaic")
    args = ap.parse_args(argv)

    ds = Scene2Dataset.discover("scene2")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    caps = {c.name: cv2.VideoCapture(str(c.video)) for c in ds.cameras}
    for f in [int(x) for x in args.frames.split(",") if x.strip()]:
        tiles = []
        for name in ds.names:
            cap = caps[name]
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                img = np.zeros((1080, 1920, 3), np.uint8)
            tile = cv2.resize(img, (640, 360))
            cv2.putText(tile, f"{name} f{f}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2, cv2.LINE_AA)
            tiles.append(tile)
        rows = [np.hstack(tiles[i * 3:(i + 1) * 3]) for i in range(3)]
        mosaic = np.vstack(rows)
        p = out / f"mosaic_{f:04d}.png"
        cv2.imwrite(str(p), mosaic)
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
