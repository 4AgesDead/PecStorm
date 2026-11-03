"""Импорт датасета «Пеленга»: проверка файлов, manifest, отчёт о расхождениях.

Запуск:
    python apps/import_truck_dataset.py [--root huniashahatona/zadania]
        [--manifest datasets/truck/dataset_manifest.json] [--no-hashes]

Что делает:
* находит пять MP4 и пять camera CSV (``TruckDataset.discover``);
* проверяет одинаковые fps/размер кадра и общий диапазон кадров;
* пишет manifest (свойства видео, диапазоны CSV, sha256, допущения);
* печатает отчёт ``mismatch_report``: расхождение числа кадров видео и строк
  CSV (§7.1 п.9 брифа), битые ячейки GT и список инвалидных кадров.

Исходные файлы только читаются; ничего не перезаписывается.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from multiview3d.truck_dataset import DEFAULT_REPAIRS, DEFAULT_ROOT, TruckDataset


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Импорт и проверка датасета «Пеленга»")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--repairs", default=str(DEFAULT_REPAIRS))
    ap.add_argument("--manifest", default="datasets/truck/dataset_manifest.json")
    ap.add_argument("--no-hashes", action="store_true",
                    help="не считать sha256 (быстро; для больших наборов)")
    ap.add_argument("--json", default=None, help="сохранить отчёт о расхождениях в JSON")
    args = ap.parse_args(argv)

    ds = TruckDataset.discover(args.root, repairs_path=args.repairs or None)
    rep = ds.mismatch_report()

    print(f"Корень: {ds.root}")
    print(f"fps={ds.fps:g}, кадр {ds.width}x{ds.height}, "
          f"общий диапазон кадров {rep['common_range'][0]}..{rep['common_range'][1]} "
          f"({rep['common_frames']} кадров)")
    print("Камеры:")
    for cam in ds.cameras:
        info = rep["per_camera"][cam.name]
        static = "статика" if ds.provider.is_static(cam.name) else "движется"
        print(f"  {cam.name}: видео {info['video_frames']} кадров "
              f"(последний индекс {info['video_last_index']}), "
              f"CSV {info['csv_frames']} строк ({info['csv_first']}..{info['csv_last']}), "
              f"строк без видеокадра: {info['csv_rows_without_video']}, {static}")
        print(f"         {cam.video.name} / {cam.csv.name}")
    gt = ds.ground_truth()
    print(f"GT: {rep['gt_frames']} строк, кадры {rep['gt_first']}..{rep['gt_last']}, "
          f"битых ячеек: {len(gt.invalid_cells)}, репаров применено: {len(gt.repaired_frames)}")
    for cell in gt.invalid_cells:
        print(f"  инвалидный кадр {cell['frame']}: строка {cell['line']}, "
              f"{cell['column']} = {cell['raw_value']!r} -> кадр исключён из GT")
    if gt.invalid_cells:
        print(f"  кадры без GT: {sorted({c['frame'] for c in gt.invalid_cells})}")
    if not gt.repaired_frames:
        print("  репары (sidecar) не применены: verified_by_organizer=false")

    if not args.no_hashes or args.manifest:
        doc = ds.write_manifest(args.manifest, with_hashes=not args.no_hashes)
        print(f"manifest: {args.manifest} "
              f"(камер {len(doc['cameras'])}, допущений {len(doc['assumptions'])})")
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"отчёт о расхождениях: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
