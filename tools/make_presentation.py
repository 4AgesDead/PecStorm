"""Сборка презентации проекта: PPTX + автономный HTML.

Данные слайдов лежат в SLIDES. Числа взяты из docs/RESULTS.md,
docs/SCENE2.md и README.md.
"""

from __future__ import annotations

import base64
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]

FONT = "Segoe UI"
INK = RGBColor(0x0F, 0x17, 0x2A)
MUTED = RGBColor(0x47, 0x55, 0x69)
ACCENT = RGBColor(0x0F, 0x76, 0x6E)
ACCENT2 = RGBColor(0xD9, 0x77, 0x06)
GOOD = RGBColor(0x15, 0x80, 0x3D)
BAD = RGBColor(0xB9, 0x1C, 0x1C)
CARD_BG = RGBColor(0xF1, 0xF5, 0xF9)
BORDER = RGBColor(0xCB, 0xD5, 0xE1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

IMG_SCENE1 = ROOT / "results/truck/report_final/comparison_vs_gt.png"
IMG_SCENE1_DASH = ROOT / "out/truck/_dash_gt_mid.png"
IMG_SCENE2 = ROOT / "results/scene2/scene2_submission.png"
IMG_SCENE2_DASH = ROOT / "out/scene2/_dash_1200.png"

SLIDES: list[dict] = [
    {
        "kind": "title",
        "title": "3D-координаты объекта по камерам",
        "subtitle": "Полный конвейер: детекция → сопоставление видов → триангуляция → "
                    "трекинг → CSV",
        "meta": "2 задания · 5 и 9 камер · 1932 кадра · 92 теста",
        "notes": "Проект «Пеленга»: по видео с нескольких калиброванных камер Unreal "
                 "восстанавливаем 3D-координаты грузовика. Два задания: 5 камер с эталоном "
                 "и 9 камер без эталона. Покажу архитектуру, три главные проблемы и числа.",
    },
    {
        "kind": "two_col",
        "title": "Что дано и что нужно",
        "left_title": "Дано",
        "left": [
            (0, "scene1: 5 камер, 1001 кадр, 1920×1080, 30 fps, есть GT"),
            (0, "scene2: 9 камер, 1932 кадра, GT нет"),
            (0, "CSV поз Unreal: CP1251, сантиметры, Y-up, фокус в мм"),
            (0, "Одна из камер scene2 движется (аэрокамера)"),
        ],
        "right_title": "Нужно",
        "right": [
            (0, "x, y, z объекта на каждый кадр"),
            (0, "Один CSV на задание, координаты отдельными столбцами"),
            (0, "Без ручной разметки кадров"),
            (0, "Понятный статус: измерение или интерполяция"),
        ],
        "callout": "Главная сложность: центр bbox детектора — не 3D-точка объекта.",
        "notes": "Дано: видео и калибровка. Нужно: координаты на каждый кадр. Ключевой "
                 "источник ошибки — bbox детектора описывает кузов, а не точку объекта.",
    },
    {
        "kind": "pipeline",
        "title": "Конвейер: от кадров до CSV",
        "steps": [
            ("Позы камер", "CSV Unreal → матрицы R, t на кадр"),
            ("Детекция", "YOLOv8, класс truck, 9 камер"),
            ("Виды", "сопоставление по предсказанию"),
            ("3D", "DLT + LM, порог 50 px"),
            ("Трекинг", "Калман, гейт 10σ"),
            ("Экспорт", "CSV, метрики, видео"),
        ],
        "callout": "Один конвейер на 5 и на 9 камер: отличаются данные и пороги.",
        "notes": "Шесть шагов. Позы читаются на каждый кадр, потому что камеры могут "
                 "двигаться. Триангуляция уточняется Левенбергом-Марквардтом, трек "
                 "держится фильтром Калмана.",
    },
    {
        "kind": "kpi",
        "title": "Проблема №1: центр bbox — не пивот объекта",
        "bullets": [
            (0, "Измерено расхождение «центр bbox ↔ проекция GT-точки»: 57–229 px по камерам"),
            (0, "Причина: bbox — середина кузова, GT — pivot актора; сдвиг зависит от ракурса"),
            (0, "Решение: 3D-сдвиг по наборам видов + поправка по азимуту"),
        ],
        "cards": [
            ("140 px", "медиана сдвига", None),
            ("2.02 м", "MAE до поправки", None),
            ("1.48 м", "MAE после азимута", GOOD),
        ],
        "callout": "Постоянный 3D-сдвиг и 2D-сдвиг по камерам проверили и отвергли: "
                   "не переносятся на камеру без данных.",
        "notes": "Это главная причина, почему «просто триангуляция» даёт метры ошибки. "
                 "Сдвиг зависит от того, какие камеры видят объект и под каким углом.",
    },
    {
        "kind": "kpi",
        "title": "Проблема №2: трек рвался почти каждый кадр",
        "bullets": [
            (0, "При gate 3σ трек пересоздавался: 486 треков на 780 точек"),
            (0, "48 % точек имели нулевую скорость — трекер не успевал разогнаться"),
            (0, "Свип порогов: gate 10σ фрагментацию убирает, MAE измерений не меняет"),
        ],
        "cards": [
            ("486 → 49", "треков", GOOD),
            ("5.35 → 4.05 м", "ошибка прогноза", GOOD),
            ("0", "потерь в MAE", None),
        ],
        "callout": "Порог выбран свипом на отложенных кадрах, а не на глаз.",
        "notes": "Фрагментация трека — вторая по важности проблема: измерений столько же, "
                 "но прогнозы в пропусках становятся заметно точнее.",
    },
    {
        "kind": "bullets",
        "title": "Детектор: COCO → псевдо-разметка → ансамбль",
        "bullets": [
            (0, "База: COCO yolov8x — MAE 4.28 м, покрытие 61.3 %"),
            (0, "Псевдо-разметка из GT и 3D-модели CargoTruck.fbx: 3077 боксов, IoU 0.64"),
            (0, "Дообучены v1 (yolov8s), v2 (yolov8m), v4b (fine-tune на своих кадрах)"),
            (0, "Финал: ансамбль v1 + v2 + v4b, слияние боксов по IoU"),
            (1, "5005 камеро-кадров: 53 мин CPU / 3.5 мин DirectML / 1 мин GPU"),
            (1, "yolo26x на наших данных не лучше yolov8x: 30.5 % против 31.4 % detection rate"),
        ],
        "callout": "Дообучение на псевдо-разметке дало основной прирост точности детекции.",
        "notes": "Модель CargoTruck.fbx использовали как источник разметки: подгоняли её "
                 "проекцию к детекциям и получали боксы для обучения.",
    },
    {
        "kind": "table",
        "title": "Результат задания 1 (5 камер, есть эталон)",
        "head": ["метрика", "COCO yolov8x", "финал v1+v2+v4b"],
        "rows": [
            ["MAE по измеренным кадрам", "4.28 м", "1.98 м"],
            ["медиана ошибки", "2.91 м", "1.42 м"],
            ["покрытие кадров", "61.3 %", "82.3 %"],
            ["holdout 700..1000", "—", "1.98 м / 67.4 %"],
            ["recall-прокси (пивот в bbox)", "0.377", "0.676"],
            ["precision-прокси", "0.775", "0.790"],
        ],
        "highlight_col": 2,
        "callout": "MAE считается только по кадрам, где измерение получено; holdout — "
                   "отдельная цифра, чтобы видеть перенос.",
        "notes": "1.98 м — это MAE по 82.3 % кадров. На отложенных кадрах 700..1000 — "
                 "те же 1.98 м при покрытии 67.4 %. Потолок с идеальными боксами — 95.5 %.",
    },
    {
        "kind": "image",
        "title": "Задание 1: оценка против эталона",
        "image": IMG_SCENE1,
        "caption": "Слева — траектория в плане (цвет = ошибка), справа — ошибка по кадрам. "
                   "823 кадра MEASURED, MAE 1.98 м.",
        "notes": "Видно, что ошибка равномерна по маршруту; всплески — участки, где объект "
                 "виден одной камерой.",
    },
    {
        "kind": "image",
        "title": "Веб-пульт: трекинг в реальном времени",
        "image": IMG_SCENE1_DASH,
        "caption": "5 камер + план X–Y + HUD: покрытие, MAE, статус кадра. "
                   "9.2–9.5 fps на одной модели v4b.",
        "notes": "Пульт на стандартной библиотеке: MJPEG-потоки, пауза, изменение порога, "
                 "JSON-статистика. Эталон рисуется только для контроля.",
    },
    {
        "kind": "bullets",
        "title": "Задание 2: 9 камер и ни одного эталона",
        "bullets": [
            (0, "9 камер, 1932 кадра; cam2 — движущаяся аэрокамера, наклон ≈ −45°"),
            (0, "GT нет: критерий качества — self-consistency, а не MAE"),
            (1, "невязка репроекции триангуляции"),
            (1, "связность трека по времени и правдоподобие скоростей"),
            (0, "CSV обязан быть на все кадры: где нет измерения — интерполяция с явным статусом"),
            (0, "Тот же конвейер, что в задании 1, плюс отдельный шаг заполнения пропусков"),
        ],
        "callout": "Фальшивые MAE и «покрытие против эталона» не публикуем: эталона нет.",
        "notes": "Здесь важно не выдать интерполяцию за измерение. В CSV два статуса: "
                 "MEASURED и INTERPOLATED.",
    },
    {
        "kind": "kpi",
        "title": "Найденная ошибка калибровки",
        "bullets": [
            (0, "Матрица поворота камеры была не ортонормальной: det ≠ 1"),
            (0, "При малых углах наклона две ошибки компенсировали друг друга — 8 камер работали"),
            (0, "У аэрокамеры cam2 (pitch ≈ −45°) det ≈ 0.08: проекция неверна, измерений нет"),
            (0, "После исправления det = 1 у всех 9 камер scene2"),
        ],
        "cards": [
            ("0.08 → 1.0", "det матрицы cam2", GOOD),
            ("45 → 113", "измерений на тесте 350..650", GOOD),
            ("499 → 703", "измерений на всём клипе", GOOD),
        ],
        "callout": "Задание 1 оставили на прежней формуле: его якоря подгонялись под неё.",
        "notes": "Это самый ценный инженерный результат второго задания: ошибка была не в "
                 "детекторе, а в геометрии. Исправление включило движущуюся аэрокамеру.",
    },
    {
        "kind": "table",
        "title": "Результат задания 2 (9 камер, без эталона)",
        "head": ["показатель", "было", "стало"],
        "rows": [
            ["кадров MEASURED", "499", "703"],
            ["кадров INTERPOLATED", "1433", "1229"],
            ["длина пути X–Y", "457 м", "582 м"],
            ["невязка репроекции, медиана", "—", "3.3 px"],
            ["дрожание траектории, медиана", "—", "0.24 м"],
            ["камер в измерениях", "5", "9"],
        ],
        "highlight_col": 2,
        "callout": "Проверка без эталона: проекция точки на аэрокадр cam2 попадает в "
                   "грузовик (кадры 50, 100, 140, 470, 520, 560).",
        "notes": "Рост не за счёт «дописывания» точек: каждая новая точка — триангуляция "
                 "минимум по двум камерам с невязкой ниже 50 px.",
    },
    {
        "kind": "image",
        "title": "Задание 2: траектория на 1932 кадра",
        "image": IMG_SCENE2,
        "caption": "703 измерения (оранжевые) и интерполяция (зелёная линия). "
                   "Маршрут ≈ 582 м.",
        "notes": "Длинные прямые участки — там, где объект виден одной камерой; "
                 "на них положение достраивается между измерениями.",
    },
    {
        "kind": "image",
        "title": "Задание 2: видео-отчёт по 9 камерам",
        "image": IMG_SCENE2_DASH,
        "caption": "9 камер + план X–Y + таблица статусов. GT не рисуется: его нет.",
        "notes": "Видео собирается тем же кодом, что и пульт, но без эталона и без "
                 "выдуманных метрик.",
    },
    {
        "kind": "bullets",
        "title": "Чем проверено и что отвергнуто",
        "bullets": [
            (0, "Отложенные кадры 700..1000 и parity-CV (чётные/нечётные) — не только in-sample"),
            (0, "92 unit-теста, синтетика с известной геометрией: ошибка 10.1 мм"),
            (0, "Отвергнуто и зафиксировано в документации:"),
            (1, "постоянный 3D-сдвиг и 2D-сдвиг по камерам — не переносятся"),
            (1, "TTA, hard negatives, покадровое уточнение меток — без выигрыша"),
            (1, "scene2: дообучение детектора (23 против 45), fit модели (эффекта нет), "
                "single-view (5.0 м ошибки)"),
        ],
        "callout": "Отрицательные результаты — часть результата: они не попали в финальные числа.",
        "notes": "Мы отдельно фиксируем, что не сработало, чтобы не выдавать это за успех.",
    },
    {
        "kind": "bullets",
        "title": "Ограничения",
        "bullets": [
            (0, "Единицы мира не подтверждены: подгонка модели даёт масштаб ≈ 3.06"),
            (0, "Синхронность камер — по номеру кадра, PTS недоступен"),
            (0, "Реальный пульт: 9.2–9.5 fps на 5 камер, 30 fps не держим"),
            (0, "Покрытие упирается в recall детектора 0.62 при нужных ~ 0.9"),
            (0, "Все числа — на одном клипе; на скрытом наборе выигрыш будет меньше"),
            (0, "В задании 2 эталона нет: MAE не считается в принципе"),
        ],
        "callout": "Все оговорки и открытые вопросы — в docs/OPEN_QUESTIONS.md и docs/RESULTS.md.",
        "notes": "Честные ограничения: не обещаем 30 fps и не называем интерполяцию "
                 "измерением.",
    },
    {
        "kind": "bullets",
        "title": "Итог и артефакты",
        "bullets": [
            (0, "Рабочий конвейер на два задания: 5 камер с эталоном и 9 камер без него"),
            (0, "Задание 1: MAE 1.98 м при покрытии 82.3 %, holdout — 1.98 м / 67.4 %"),
            (0, "Задание 2: CSV на все 1932 кадра, 703 измерения, видео-отчёт"),
            (0, "Репозиторий: код, тесты, документация, инструкция по запуску"),
            (0, "Воспроизводимость: py -m unittest discover -s tests → 92 теста, OK"),
        ],
        "callout": "Готово к передаче: код, данные, веса и инструкция лежат в одной папке.",
        "notes": "Финал: два задания закрыты, числа воспроизводятся командами из инструкции.",
    },
]


KICKERS = {
    2: "Задача",
    3: "Конвейер",
    4: "Проблема",
    5: "Проблема",
    6: "Детектор",
    7: "Результат",
    8: "Результат",
    9: "Интерфейс",
    10: "Задание 2",
    11: "Задание 2",
    12: "Результат",
    13: "Результат",
    14: "Результат",
    15: "Проверки",
    16: "Ограничения",
    17: "Итог",
}


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def add_text(slide, x, y, w, h, text, size=18, bold=False, color=INK,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.1):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = spacing
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = FONT
    return tb


def add_bullets(slide, x, y, w, h, items, size=17, gap=9):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    first = True
    for level, text in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap if level == 0 else max(2, gap - 5))
        p.line_spacing = 1.14
        marker = "•  " if level == 0 else "–  "
        run = p.add_run()
        run.text = marker + text
        run.font.size = Pt(size if level == 0 else size - 2)
        run.font.color.rgb = INK if level == 0 else MUTED
        run.font.name = FONT
        marl = Inches(0.30 if level == 0 else 0.62)
        indent = -Inches(0.30 if level == 0 else 0.28)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(marl)))
        pPr.set("indent", str(int(indent)))


def add_footer(slide, number: int, total: int):
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(6.98),
                                  Inches(11.93), Pt(0.75))
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER
    line.line.fill.background()
    line.shadow.inherit = False
    add_text(slide, 0.7, 7.08, 7.0, 0.3,
             "Пеленга · 3D-координаты по камерам", size=10, color=MUTED)
    add_text(slide, 10.2, 7.08, 2.43, 0.3, f"{number} / {total}", size=10,
             color=MUTED, align=PP_ALIGN.RIGHT)


def add_title(slide, text: str, kicker: str | None = None):
    y = 0.5
    if kicker:
        add_text(slide, 0.7, 0.42, 11.9, 0.3, kicker.upper(), size=10.5,
                 bold=True, color=ACCENT)
        y = 0.72
    add_text(slide, 0.7, y, 11.9, 0.7, text, size=28, bold=True, color=INK)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(1.42),
                                 Inches(1.1), Inches(0.055))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    bar.shadow.inherit = False


def add_callout(slide, text: str, y: float, color: RGBColor = ACCENT):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(y),
                                 Inches(11.93), Inches(0.72))
    box.fill.solid()
    box.fill.fore_color.rgb = CARD_BG
    box.line.color.rgb = BORDER
    box.line.width = Pt(1)
    box.shadow.inherit = False
    box.adjustments[0] = 0.08
    strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(y),
                                   Inches(0.06), Inches(0.72))
    strip.fill.solid()
    strip.fill.fore_color.rgb = color
    strip.line.fill.background()
    strip.shadow.inherit = False
    add_text(slide, 0.95, y + 0.08, 11.5, 0.56, text, size=13, color=INK,
             anchor=MSO_ANCHOR.MIDDLE)


def add_cards(slide, cards: list[tuple[str, str, RGBColor | None]], y: float):
    n = len(cards)
    gap = 0.3
    w = (11.93 - gap * (n - 1)) / n
    for i, (big, label, color) in enumerate(cards):
        x = 0.7 + i * (w + gap)
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                     Inches(w), Inches(1.45))
        box.fill.solid()
        box.fill.fore_color.rgb = CARD_BG
        box.line.color.rgb = BORDER
        box.line.width = Pt(1)
        box.shadow.inherit = False
        box.adjustments[0] = 0.08
        add_text(slide, x + 0.25, y + 0.16, w - 0.5, 0.6, big, size=26, bold=True,
                 color=color or INK)
        add_text(slide, x + 0.25, y + 0.88, w - 0.5, 0.45, label, size=11.5, color=MUTED)


def add_table(slide, head: list[str], rows: list[list[str]], highlight_col: int, y: float):
    n_rows = len(rows) + 1
    n_cols = len(head)
    height = 0.46 * n_rows
    table = slide.shapes.add_table(n_rows, n_cols, Inches(0.7), Inches(y),
                                   Inches(11.93), Inches(height)).table
    widths = [6.2] + [2.86] * (n_cols - 1)
    for i, w in enumerate(widths[:n_cols]):
        table.columns[i].width = Inches(w)
    table.rows[0].height = Inches(0.5)
    for r in range(1, n_rows):
        table.rows[r].height = Inches(0.44)
    for c, text in enumerate(head):
        cell = table.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = INK
        cell.margin_left = Inches(0.16)
        cell.margin_right = Inches(0.1)
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
        run = p.add_run()
        run.text = text
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = WHITE
        run.font.name = FONT
    for r, row in enumerate(rows, start=1):
        for c, text in enumerate(row):
            cell = table.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if r % 2 else CARD_BG
            cell.margin_left = Inches(0.16)
            cell.margin_right = Inches(0.1)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = text
            run.font.size = Pt(13)
            run.font.name = FONT
            if c == highlight_col:
                run.font.bold = True
                run.font.color.rgb = GOOD
            else:
                run.font.color.rgb = INK
    return height


def add_image_fit(slide, path: Path, x: float, y: float, w: float, h: float):
    iw, ih = image_size(path)
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    slide.shapes.add_picture(str(path), Inches(x + (w - dw) / 2),
                             Inches(y + (h - dh) / 2), Inches(dw), Inches(dh))


def build_pptx(out: Path) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    total = len(SLIDES)
    for idx, spec in enumerate(SLIDES, start=1):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        kind = spec["kind"]
        if kind == "title":
            band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                                          Inches(0.22), Inches(7.5))
            band.fill.solid()
            band.fill.fore_color.rgb = ACCENT
            band.line.fill.background()
            band.shadow.inherit = False
            add_text(slide, 0.9, 1.85, 11.5, 0.4, "ПЕЛЕНГА · ХАКАТОН", size=12,
                     bold=True, color=ACCENT)
            add_text(slide, 0.9, 2.3, 11.5, 1.6, spec["title"], size=44, bold=True,
                     color=INK)
            add_text(slide, 0.9, 3.95, 10.6, 0.9, spec["subtitle"], size=18, color=MUTED)
            rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(5.05),
                                          Inches(1.6), Inches(0.06))
            rule.fill.solid()
            rule.fill.fore_color.rgb = ACCENT
            rule.line.fill.background()
            rule.shadow.inherit = False
            add_text(slide, 0.9, 5.3, 11.5, 0.5, spec["meta"], size=14, color=INK)
        else:
            add_title(slide, spec["title"], spec.get("kicker") or KICKERS.get(idx))
            if kind == "bullets":
                add_bullets(slide, 0.7, 1.75, 11.6, 3.7, spec["bullets"])
                if spec.get("callout"):
                    add_callout(slide, spec["callout"], 5.55)
            elif kind == "kpi":
                add_bullets(slide, 0.7, 1.72, 11.6, 2.1, spec["bullets"])
                add_cards(slide, spec["cards"], 3.95)
                if spec.get("callout"):
                    add_callout(slide, spec["callout"], 5.75)
            elif kind == "table":
                h = add_table(slide, spec["head"], spec["rows"],
                              spec.get("highlight_col", 1), 1.8)
                if spec.get("callout"):
                    add_callout(slide, spec["callout"], 1.9 + h + 0.25)
            elif kind == "image":
                add_image_fit(slide, spec["image"], 0.7, 1.65, 11.93, 4.5)
                add_text(slide, 0.7, 6.32, 11.93, 0.5, spec["caption"], size=12,
                         color=MUTED)
            elif kind == "two_col":
                add_text(slide, 0.7, 1.75, 5.6, 0.4, spec["left_title"], size=15,
                         bold=True, color=ACCENT)
                add_bullets(slide, 0.7, 2.2, 5.6, 2.6, spec["left"], size=15)
                add_text(slide, 6.9, 1.75, 5.7, 0.4, spec["right_title"], size=15,
                         bold=True, color=ACCENT)
                add_bullets(slide, 6.9, 2.2, 5.7, 2.6, spec["right"], size=15)
                add_callout(slide, spec["callout"], 5.55)
            elif kind == "pipeline":
                n = len(spec["steps"])
                gap = 0.28
                bw = (11.93 - gap * (n - 1)) / n
                for i, (name, desc) in enumerate(spec["steps"]):
                    x = 0.7 + i * (bw + gap)
                    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                                 Inches(x), Inches(2.15),
                                                 Inches(bw), Inches(2.05))
                    box.fill.solid()
                    box.fill.fore_color.rgb = CARD_BG
                    box.line.color.rgb = BORDER
                    box.line.width = Pt(1)
                    box.shadow.inherit = False
                    box.adjustments[0] = 0.08
                    num = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.18),
                                                 Inches(2.3), Inches(0.42), Inches(0.42))
                    num.fill.solid()
                    num.fill.fore_color.rgb = ACCENT
                    num.line.fill.background()
                    num.shadow.inherit = False
                    add_text(slide, x + 0.18, 2.33, 0.42, 0.36, str(i + 1), size=13,
                             bold=True, color=WHITE, align=PP_ALIGN.CENTER)
                    add_text(slide, x + 0.18, 2.88, bw - 0.36, 0.5, name, size=13.5,
                             bold=True, color=INK)
                    add_text(slide, x + 0.18, 3.36, bw - 0.36, 0.75, desc, size=10.5,
                             color=MUTED)
                    if i < n - 1:
                        add_text(slide, x + bw - 0.02, 2.95, gap + 0.04, 0.4, "›",
                                 size=18, bold=True, color=ACCENT,
                                 align=PP_ALIGN.CENTER)
                add_callout(slide, spec["callout"], 4.75)
        if spec.get("notes"):
            slide.notes_slide.notes_text_frame.text = spec["notes"]
        add_footer(slide, idx, total)
    prs.save(str(out))
    print(f"PPTX: {out}")


CSS = """
:root { --ink:#0f172a; --muted:#475569; --accent:#0f766e; --card:#f1f5f9; --border:#cbd5e1; }
* { box-sizing: border-box; }
body { margin:0; background:#0b1220; font-family:"Segoe UI",Arial,sans-serif; color:var(--ink); }
.deck { width:1280px; margin:0 auto; }
.slide { width:1280px; height:720px; background:#fff; padding:56px 64px 40px; position:relative;
         display:none; flex-direction:column; }
.slide.active { display:flex; }
.kicker { color:var(--accent); font-size:13px; font-weight:700; letter-spacing:.14em;
          text-transform:uppercase; margin-bottom:10px; }
h1 { font-size:44px; margin:0 0 14px; line-height:1.12; }
h2 { font-size:30px; margin:0 0 6px; line-height:1.15; }
.rule { width:72px; height:4px; background:var(--accent); margin:12px 0 26px; }
p.lead { font-size:20px; color:var(--muted); margin:0; max-width:900px; }
ul { list-style:none; padding:0; margin:0; font-size:17px; line-height:1.42; }
li { margin-bottom:10px; padding-left:24px; position:relative; }
li:before { content:"•"; position:absolute; left:4px; color:var(--accent); }
li.sub { font-size:15px; color:var(--muted); margin-left:22px; }
li.sub:before { content:"–"; }
.cards { display:flex; gap:18px; margin:22px 0; }
.card { flex:1; background:var(--card); border:1px solid var(--border); border-radius:12px;
        padding:16px 18px; }
.card b { display:block; font-size:30px; line-height:1.1; }
.card span { color:var(--muted); font-size:13px; }
.good b { color:#15803d; }
.callout { background:var(--card); border-left:4px solid var(--accent); border-radius:8px;
           padding:12px 16px; font-size:15px; margin-top:auto; }
table { border-collapse:collapse; width:100%; font-size:15px; margin-top:6px; }
th { background:var(--ink); color:#fff; text-align:left; padding:10px 12px; font-weight:600; }
td { padding:9px 12px; border-bottom:1px solid #e2e8f0; }
tr:nth-child(even) td { background:#f8fafc; }
td.hl { color:#15803d; font-weight:700; }
.shot { flex:1; display:flex; align-items:center; justify-content:center; overflow:hidden; }
.shot img { max-width:100%; max-height:430px; object-fit:contain; }
.cap { color:var(--muted); font-size:13px; margin-top:12px; }
.foot { position:absolute; left:64px; right:64px; bottom:18px; display:flex;
        justify-content:space-between; color:var(--muted); font-size:11px;
        border-top:1px solid #e2e8f0; padding-top:8px; }
.cols { display:flex; gap:48px; }
.cols > div { flex:1; }
.cols h3 { color:var(--accent); font-size:16px; margin:0 0 12px; }
.steps { display:flex; gap:10px; margin-top:10px; }
.step { flex:1; background:var(--card); border:1px solid var(--border); border-radius:12px;
        padding:14px; }
.step i { display:inline-flex; width:26px; height:26px; border-radius:50%; background:var(--accent);
          color:#fff; font-style:normal; font-size:13px; align-items:center; justify-content:center;
          margin-bottom:10px; }
.step b { display:block; font-size:14px; margin-bottom:6px; }
.step span { font-size:12px; color:var(--muted); }
.hint { position:fixed; right:18px; bottom:14px; color:#94a3b8; font-size:12px; }
@media print { body{background:#fff} .slide{display:flex !important; page-break-after:always}
               .hint{display:none} }
"""

JS = """
const slides=[...document.querySelectorAll('.slide')];
let i=0;
function render(n){ i=Math.max(0,Math.min(slides.length-1,n));
  slides.forEach((s,k)=>s.classList.toggle('active',k===i)); }
function go(n){ render(n); history.replaceState(null,'','#'+(i+1)); }
document.addEventListener('keydown',e=>{
  if(['ArrowRight','ArrowDown',' ','PageDown'].includes(e.key)){e.preventDefault();go(i+1);}
  if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)){e.preventDefault();go(i-1);}
  if(e.key==='Home')go(0); if(e.key==='End')go(slides.length-1);
});
window.addEventListener('hashchange',()=>{
  const n=parseInt(location.hash.replace('#',''),10)-1;
  if(!isNaN(n)) render(n);
});
const start=parseInt(location.hash.replace('#',''),10)-1;
render(isNaN(start)?0:start);
"""


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def bullets_html(items) -> str:
    out = []
    for level, text in items:
        cls = ' class="sub"' if level else ""
        out.append(f"<li{cls}>{esc(text)}</li>")
    return "<ul>" + "".join(out) + "</ul>"


def img_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{data}"


def build_html(out: Path) -> None:
    parts = ['<!doctype html><html lang="ru"><head><meta charset="utf-8">',
             "<title>Пеленга · презентация</title>", f"<style>{CSS}</style></head><body>",
             '<div class="deck">']
    total = len(SLIDES)
    for idx, spec in enumerate(SLIDES, start=1):
        kind = spec["kind"]
        parts.append('<section class="slide">')
        if kind == "title":
            parts.append('<div class="kicker">Пеленга · хакатон</div>')
            parts.append(f"<h1>{esc(spec['title'])}</h1>")
            parts.append(f'<p class="lead">{esc(spec["subtitle"])}</p>')
            parts.append('<div style="height:26px"></div><div class="rule"></div>')
            parts.append(f'<p class="lead" style="font-size:16px">{esc(spec["meta"])}</p>')
        else:
            kicker = spec.get("kicker") or KICKERS.get(idx)
            if kicker:
                parts.append(f'<div class="kicker">{esc(kicker)}</div>')
            parts.append(f'<h2>{esc(spec["title"])}</h2><div class="rule"></div>')
            if kind == "bullets":
                parts.append(bullets_html(spec["bullets"]))
                if spec.get("callout"):
                    parts.append(f'<div class="callout">{esc(spec["callout"])}</div>')
            elif kind == "kpi":
                parts.append(bullets_html(spec["bullets"]))
                parts.append('<div class="cards">')
                for big, label, color in spec["cards"]:
                    cls = "card good" if color else "card"
                    parts.append(f'<div class="{cls}"><b>{esc(big)}</b>'
                                 f"<span>{esc(label)}</span></div>")
                parts.append("</div>")
                if spec.get("callout"):
                    parts.append(f'<div class="callout">{esc(spec["callout"])}</div>')
            elif kind == "table":
                parts.append("<table><thead><tr>")
                for c in spec["head"]:
                    parts.append(f"<th>{esc(c)}</th>")
                parts.append("</tr></thead><tbody>")
                for row in spec["rows"]:
                    parts.append("<tr>")
                    for j, cell in enumerate(row):
                        cls = ' class="hl"' if j == spec.get("highlight_col", 1) else ""
                        parts.append(f"<td{cls}>{esc(cell)}</td>")
                    parts.append("</tr>")
                parts.append("</tbody></table>")
                if spec.get("callout"):
                    parts.append(f'<div class="callout">{esc(spec["callout"])}</div>')
            elif kind == "image":
                parts.append('<div class="shot">')
                parts.append(f'<img src="{img_data_uri(spec["image"])}" alt="">')
                parts.append("</div>")
                parts.append(f'<div class="cap">{esc(spec["caption"])}</div>')
            elif kind == "two_col":
                parts.append('<div class="cols"><div>')
                parts.append(f"<h3>{esc(spec['left_title'])}</h3>")
                parts.append(bullets_html(spec["left"]))
                parts.append("</div><div>")
                parts.append(f"<h3>{esc(spec['right_title'])}</h3>")
                parts.append(bullets_html(spec["right"]))
                parts.append("</div></div>")
                parts.append(f'<div class="callout">{esc(spec["callout"])}</div>')
            elif kind == "pipeline":
                parts.append('<div class="steps">')
                for i, (name, desc) in enumerate(spec["steps"], start=1):
                    parts.append(f'<div class="step"><i>{i}</i>'
                                 f"<b>{esc(name)}</b><span>{esc(desc)}</span></div>")
                parts.append("</div>")
                parts.append(f'<div class="callout">{esc(spec["callout"])}</div>')
        parts.append(f'<div class="foot"><span>Пеленга · 3D-координаты по камерам</span>'
                     f"<span>{idx} / {total}</span></div>")
        parts.append("</section>")
    parts.append("</div><div class=\"hint\">→ / ← — слайды</div>")
    parts.append(f"<script>{JS}</script></body></html>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    print(f"HTML: {out}")


def main() -> int:
    pptx_path = ROOT / "ПРЕЗЕНТАЦИЯ.pptx"
    html_path = ROOT / "presentation" / "презентация.html"
    build_pptx(pptx_path)
    build_html(html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
