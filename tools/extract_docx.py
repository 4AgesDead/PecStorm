"""Конвертация .docx в markdown: только текст, форматирование упрощено.

Запуск: ``python tools/extract_docx.py <in.docx> <out.md>``
Извлечение из ``word/document.xml``: абзацы, заголовки по стилям ``HeadingN``,
списки (``numPr``/``ListParagraph``) — маркированные, таблицы — markdown.
Изображения и колонтитулы не переносятся (их число печатается в шапке вывода).
"""

import hashlib
import sys
import zipfile
from datetime import date
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def para_text(p) -> str:
    return "".join(t.text or "" for t in p.iter(W + "t")).strip()


def para_style(p) -> str:
    ppr = p.find(W + "pPr")
    if ppr is None:
        return ""
    st = ppr.find(W + "pStyle")
    return st.get(W + "val") if st is not None else ""


def is_list(p) -> bool:
    ppr = p.find(W + "pPr")
    return ppr is not None and ppr.find(W + "numPr") is not None


def table_md(tbl) -> str:
    rows = []
    for tr in tbl.findall(W + "tr"):
        cells = [para_text(tc).replace("|", "\\|") for tc in tr.findall(W + "tc")]
        rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * width) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def para_md(p) -> str:
    text = para_text(p)
    if not text:
        return ""
    style = para_style(p)
    head = ""
    if style.lower().startswith("heading") and style[len("heading"):].isdigit():
        head = "#" * int(style[len("heading"):]) + " "
    elif style == "Title":
        head = "# "
    if head or not (is_list(p) or style == "ListParagraph"):
        return head + text
    return "- " + text


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    data = src.read_bytes()
    z = zipfile.ZipFile(src)
    media = [n for n in z.namelist() if n.startswith("word/media/")]
    root = ET.fromstring(z.read("word/document.xml"))
    body = root.find(W + "body")
    blocks = []
    for child in body:
        if child.tag == W + "p":
            md = para_md(child)
            if md:
                blocks.append(md)
        elif child.tag == W + "tbl":
            md = table_md(child)
            if md:
                blocks.append(md)
    sha = hashlib.sha256(data).hexdigest()
    header = (
        "# ТЗ хакатон 10.2026 — полный текст оригинала\n\n"
        f"> Оригинал: `{src}`\n"
        f"> ({len(data)} байт, sha256 `{sha}`).\n"
        f"> Скопирован в репозиторий {date.today().isoformat()} скриптом "
        "`tools/extract_docx.py` (извлечение текста из `word/document.xml`).\n"
        "> Форматирование упрощено: заголовки — по стилям HeadingN, списки — "
        "маркированные (нумерация оригинала может отличаться), таблицы — markdown. "
        f"Изображений в документе: {len(media)} (не перенесены)."
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(header + "\n\n" + "\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"{dst}: абзацев/блоков {len(blocks)}, sha256 оригинала {sha}, media {len(media)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
