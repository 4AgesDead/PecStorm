"""Единая команда проекта: `pecstorm <команда> [аргументы]`.

Запускает стадии конвейера, приложения и сервисы, не требуя помнить пути к
скриптам и интерпретаторам. Для запуска из cmd в корне репозитория лежит
`pecstorm.cmd` (он же добавляет корень в PATH при необходимости).

Примеры:
    pecstorm --help
    pecstorm launch                 # веб-пульт на 127.0.0.1:8000
    pecstorm launch --stop          # остановить пульт
    pecstorm detect --start 0 --end 100 --step 20
    pecstorm reconstruct --detections out/truck/detections.csv
    pecstorm test
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out" / "pecstorm"
PID_FILE = OUT / "launch.pid"
LOG_FILE = OUT / "launch.log"

# команда -> (скрипт, описание)
COMMANDS: dict[str, tuple[str, str]] = {
    "import-data": ("apps/import_truck_dataset.py", "инвентаризация датасета + manifest"),
    "detect": ("apps/detect_truck.py", "детекция YOLO по кадрам (CPU/GPU/ONNX-DML)"),
    "reconstruct": ("apps/reconstruct_truck.py", "3D-реконструкция и трекинг по детекциям"),
    "evaluate": ("apps/evaluate_truck.py", "MAE/RMSE/P95 и покрытие против GT"),
    "visualize": ("apps/visualize_truck.py", "проекции оценки и GT на кадры камер"),
    "video": ("tools/render_tracking_video.py", "MP4 в стиле пульта: путь, прогноз, план X–Y"),
    "dashboard-video": ("tools/render_dashboard_video.py", "запись 5 камер + план X–Y + таблица"),
    "dashboard-report": ("tools/dashboard_report.py", "таблица прогона веб-пульта"),
    "coords-table": ("tools/make_coords_table.py", "CSV координат оценки vs эталона по пробегу"),
    "plot-gt": ("tools/plot_vs_gt.py", "график «оценка vs эталон» (GT)"),
    "report": ("tools/make_report.py", "сводка и графики по прогону"),
    "charts": ("tools/make_comparison_charts.py", "сравнительные графики конфигураций"),
    "labels": ("tools/make_pseudo_labels.py", "псевдо-разметка из GT + 3D-модели"),
    "merge": ("tools/merge_detections.py", "слияние детекций нескольких моделей"),
    "train": ("tools/train_truck_detector.py", "обучение/дообучение детектора"),
    "eval-model": ("tools/run_finetuned_eval.py", "полный цикл оценки чекпойнта"),
    "anchor-views": ("tools/anchor_fit_views.py", "якорь по наборам видов"),
    "anchor-azimuth": ("tools/anchor_fit_azimuth.py", "азимутальный якорь"),
    "smoke": ("tools/smoke_check.py", "быстрая проверка ядра"),
    "status": ("", "показать, запущен ли пульт"),
}

HELP_TABLE = "\n".join(
    f"  {name:14} {desc}" for name, (_, desc) in COMMANDS.items()
) + "\n  launch         веб-пульт с трекингом (--stop — остановить)\n" \
    "  test           все тесты unittest\n"

VENV_PY = ROOT / ".venv-rocm" / "Scripts" / "python.exe"


def python_exe() -> str:
    """Интерпретатор для стадий: venv с GPU, если есть, иначе текущий."""
    if VENV_PY.exists():
        return str(VENV_PY)
    return sys.executable


def run_script(script: str, extra: list[str]) -> int:
    path = ROOT / script
    if not path.exists():
        print(f"нет файла {path}", file=sys.stderr)
        return 2
    cmd = [python_exe(), str(path), *extra]
    return subprocess.call(cmd, cwd=str(ROOT))


def pid_alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         capture_output=True, text=True).stdout
    return str(pid) in out


def dashboard_pids() -> list[int]:
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -match 'dashboard.py' } | "
          "ForEach-Object { $_.ProcessId }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                         capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def stop_launch() -> int:
    stopped = []
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = None
        if pid and pid_alive(pid):
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True)
            stopped.append(pid)
        PID_FILE.unlink(missing_ok=True)
    for pid in dashboard_pids():
        if pid not in stopped:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True)
            stopped.append(pid)
    if stopped:
        print("остановлено: " + ", ".join(f"PID {p}" for p in stopped))
    else:
        print("пульт не запущен")
    return 0


def start_launch(extra: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    running = dashboard_pids()
    if running:
        print(f"пульт уже запущен (PID {', '.join(map(str, running))}); "
              f"сначала: pecstorm launch --stop")
        return 1
    args = extra if extra else ["--loop", "--target-fps", "15",
                                "--preview-scale", "0.5"]
    cmd = [python_exe(), str(ROOT / "apps" / "dashboard.py"), *args]
    log = LOG_FILE.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "DETACHED_PROCESS", 0)
                            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    print(f"пульт запущен (PID {proc.pid}); лог: {LOG_FILE}")
    print("открыть: http://127.0.0.1:8000/")
    print("остановить: pecstorm launch --stop")
    return 0


def launch_status() -> int:
    pids = dashboard_pids()
    if pids:
        print("пульт работает: PID " + ", ".join(map(str, pids)))
        print("адрес: http://127.0.0.1:8000/")
    else:
        print("пульт не запущен")
    return 0 if pids else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pecstorm",
        description="Единая команда проекта «Пеленга» (multiview3d).",
        epilog="Доступные команды:\n" + HELP_TABLE +
               "\nЛюбую команду можно вызвать с --help, чтобы увидеть её флаги.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd")

    p_launch = sub.add_parser("launch", add_help=False,
                              help="веб-пульт с трекингом в реальном времени")
    p_launch.add_argument("--stop", action="store_true", help="остановить пульт")
    p_launch.add_argument("-h", "--help", action="store_true", help="справка")
    p_launch.add_argument("extra", nargs=argparse.REMAINDER)

    for name in COMMANDS:
        sp = sub.add_parser(name, add_help=False, help=COMMANDS[name][1])
        sp.add_argument("-h", "--help", action="store_true")
        sp.add_argument("extra", nargs=argparse.REMAINDER)

    p_test = sub.add_parser("test", add_help=False, help="все тесты unittest")
    p_test.add_argument("-h", "--help", action="store_true")
    p_test.add_argument("extra", nargs=argparse.REMAINDER)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args, unknown = ap.parse_known_args(argv)
    if args.cmd is None:
        ap.print_help()
        return 0
    extra = list(getattr(args, "extra", []) or []) + list(unknown)

    if args.cmd == "launch":
        if getattr(args, "help", False):
            print("pecstorm launch [флаги dashboard]\n"
                  "  --stop   остановить пульт\n"
                  "  прочие флаги передаются apps/dashboard.py "
                  "(--port, --model, --device, --preview-scale, --loop ...)")
            return 0
        return stop_launch() if args.stop else start_launch(extra)
    if args.cmd == "status":
        return launch_status()
    if args.cmd == "test":
        if getattr(args, "help", False):
            print("pecstorm test — запускает `py -m unittest discover -s tests`")
            return 0
        return subprocess.call([python_exe(), "-m", "unittest", "discover", "-s", "tests"],
                               cwd=str(ROOT))
    if args.cmd in COMMANDS:
        script = COMMANDS[args.cmd][0]
        if getattr(args, "help", False):
            return run_script(script, ["--help"])
        return run_script(script, extra)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
