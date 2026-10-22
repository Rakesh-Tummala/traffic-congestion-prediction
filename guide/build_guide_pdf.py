"""Build one PDF: Part A (live demo walkthrough) + Part B (learning guide).

    python guide/build_guide_pdf.py            # writes Demo_and_Study_Guide.pdf in the project folder

Sources: DEMO_SCRIPT.md, guide/concepts.md (Part C), PROJECT_PRESENTATION_GUIDE.md (read, never modified: a few lines that are now out of
date are patched in memory), and guide/new_lessons.md (Lessons 11-15 and extra questions). Needs reportlab.
"""
import os
import re
import textwrap
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Preformatted, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "Demo_and_Study_Guide.pdf")
FONTS = r"C:\Windows\Fonts"

pdfmetrics.registerFont(TTFont("Body", os.path.join(FONTS, "segoeui.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", os.path.join(FONTS, "segoeuib.ttf")))
pdfmetrics.registerFont(TTFont("Body-Italic", os.path.join(FONTS, "segoeuii.ttf")))
pdfmetrics.registerFont(TTFont("Body-BoldItalic", os.path.join(FONTS, "segoeuiz.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic")
pdfmetrics.registerFont(TTFont("Mono", os.path.join(FONTS, "consola.ttf")))
pdfmetrics.registerFont(TTFont("Mono-Bold", os.path.join(FONTS, "consolab.ttf")))
pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-Bold", italic="Mono", boldItalic="Mono-Bold")

NAVY = colors.HexColor("#0E3A5F")
TEAL = colors.HexColor("#0E7C86")
INK = colors.HexColor("#1B2733")
MUTED = colors.HexColor("#5B6B7A")
PANEL = colors.HexColor("#EEF5F8")
LINE = colors.HexColor("#C9D6DF")

S = {
    "body": ParagraphStyle("body", fontName="Body", fontSize=10, leading=14.5, textColor=INK, spaceAfter=5, alignment=TA_LEFT),
    "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=26, leading=31, textColor=NAVY, spaceAfter=10),
    "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=17, leading=21, textColor=NAVY, spaceBefore=10, spaceAfter=7),
    "h3": ParagraphStyle("h3", fontName="Body-Bold", fontSize=12.5, leading=16, textColor=TEAL, spaceBefore=9, spaceAfter=4),
    "h4": ParagraphStyle("h4", fontName="Body-Bold", fontSize=10.5, leading=14, textColor=INK, spaceBefore=6, spaceAfter=3),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.8, leading=11.6, textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName="Body-Bold", fontSize=8.8, leading=11.6, textColor=NAVY),
    "quote": ParagraphStyle("quote", fontName="Body-Italic", fontSize=9.8, leading=14, textColor=colors.HexColor("#243B4D"), spaceAfter=4),
    "toc1": ParagraphStyle("toc1", fontName="Body-Bold", fontSize=11, leading=17, textColor=NAVY, leftIndent=0),
    "toc2": ParagraphStyle("toc2", fontName="Body", fontSize=10, leading=15, textColor=INK, leftIndent=14),
    "cover_title": ParagraphStyle("ct", fontName="Body-Bold", fontSize=34, leading=40, textColor=NAVY, spaceAfter=8),
    "cover_sub": ParagraphStyle("cs", fontName="Body", fontSize=14, leading=20, textColor=MUTED, spaceAfter=18),
}

EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u2B50\u200d]")


def clean(text: str) -> str:
    return EMOJI.sub("", text)


def inline(text: str) -> str:
    """Markdown inline (bold, italic, code) to reportlab paragraph markup."""
    text = clean(text)
    out = []
    for i, part in enumerate(re.split(r"(`[^`]+`)", text)):
        if part.startswith("`") and part.endswith("`") and len(part) > 2:
            code = part[1:-1].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            out.append(f'<font name="Mono" size="8.8" color="#0B4F6C" backColor="#E8EFF3">{code}</font>')
            continue
        part = part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        part = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", part)
        part = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", part)
        out.append(part)
    return "".join(out)


# ---------------------------------------------------------------- markdown -> blocks
LIST_RE = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")


def parse(md: str) -> list:
    lines = md.replace("\r\n", "\n").split("\n")
    blocks, i = [], 0

    def starts_block(ln: str) -> bool:
        s = ln.strip()
        return (not s or s.startswith(("```", "|", ">", "#")) or s == "---" or bool(LIST_RE.match(ln)))

    while i < len(lines):
        line, s = lines[i], lines[i].strip()
        if s.startswith("```"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            blocks.append(("code", "\n".join(code)))
        elif not s:
            i += 1
        elif re.match(r"^#{1,4}\s", s):
            level = len(s) - len(s.lstrip("#"))
            blocks.append((f"h{level}", s[level:].strip()))
            i += 1
        elif s == "---":
            blocks.append(("rule", ""))
            i += 1
        elif s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c) or not any(cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
        elif s.startswith(">"):
            paras, cur = [], []
            while i < len(lines) and lines[i].strip().startswith(">"):
                body = lines[i].strip()[1:].strip()
                if body:
                    cur.append(body)
                elif cur:
                    paras.append(" ".join(cur))
                    cur = []
                i += 1
            if cur:
                paras.append(" ".join(cur))
            blocks.append(("quote", paras))
        elif LIST_RE.match(line):
            m = LIST_RE.match(line)
            indent, marker, text = len(m.group(1).expandtabs(4)), m.group(2), m.group(3)
            i += 1
            while i < len(lines) and lines[i].strip() and not starts_block(lines[i]):
                text += " " + lines[i].strip()
                i += 1
            blocks.append(("li", (indent // 3, marker, text)))
        else:
            indent = len(line) - len(line.lstrip())
            text = s
            i += 1
            while i < len(lines) and not starts_block(lines[i]):
                text += " " + lines[i].strip()
                i += 1
            blocks.append(("p", (indent, text)))
    return blocks


# ---------------------------------------------------------------- blocks -> flowables
class Heading(Paragraph):
    def __init__(self, text, style, toc_level=None):
        super().__init__(text, style)
        self.toc_level, self.toc_text = toc_level, re.sub(r"<[^>]+>", "", text)


def make_table(rows, avail):
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    weights = []
    for c in range(ncols):
        longest = max(len(clean(r[c])) for r in rows)
        weights.append(min(max(longest, 7), 48))
    total = sum(weights)
    widths = [avail * w / total for w in weights]
    data = [[Paragraph(inline(cell), S["cellh"] if ri == 0 else S["cell"]) for cell in row] for ri, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEAF1")),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    return t


def flowables(blocks, avail, page_break_h2=False):
    out = []
    for kind, val in blocks:
        if kind == "h1":
            out.append(Heading(inline(val), S["h1"]))
        elif kind == "h2":
            if page_break_h2:
                out.append(PageBreak())
            else:
                out.append(CondPageBreak(60 * mm))
            out.append(Heading(inline(val), S["h2"], toc_level=1))
        elif kind == "h3":
            out.append(CondPageBreak(35 * mm))
            out.append(Heading(inline(val), S["h3"]))
        elif kind == "h4":
            out.append(Heading(inline(val), S["h4"]))
        elif kind == "p":
            indent, text = val
            style = S["body"] if indent < 3 else ParagraphStyle("pi", parent=S["body"], leftIndent=18)
            out.append(Paragraph(inline(text), style))
        elif kind == "li":
            level, marker, text = val
            bullet = "\u2022" if marker in "-*" else marker
            style = ParagraphStyle("li", parent=S["body"], leftIndent=16 + level * 16, bulletIndent=level * 16 + 2,
                                   bulletFontName="Body-Bold", spaceAfter=3)
            out.append(Paragraph(inline(text), style, bulletText=bullet))
        elif kind == "code":
            wrapped = "\n".join(l for line in val.split("\n") for l in (textwrap.wrap(line, 88, replace_whitespace=False,
                                                                                    drop_whitespace=False) or [""]))
            pre = Preformatted(clean(wrapped), ParagraphStyle("code", fontName="Mono", fontSize=8.4, leading=11,
                                                              textColor=colors.HexColor("#17324A")))
            box = Table([[pre]], colWidths=[avail])
            box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F8")),
                                     ("BOX", (0, 0), (-1, -1), 0.5, LINE), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                     ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
            out += [box, Spacer(1, 6)]
        elif kind == "table":
            out += [make_table(val, avail), Spacer(1, 7)]
        elif kind == "quote":
            inner = [[Paragraph(inline(p), S["quote"])] for p in val]
            box = Table(inner, colWidths=[avail - 6])
            box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PANEL), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                                     ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 2),
                                     ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
            out += [box, Spacer(1, 6)]
        elif kind == "rule":
            out.append(Spacer(1, 4))
    return out


# ---------------------------------------------------------------- sources
def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def patch_guide(g: str) -> str:
    """Bring the older guide in line with the current app, in memory only."""
    def sub(old, new):
        nonlocal g
        assert old in g, f"patch target missing: {old[:50]!r}"
        g = g.replace(old, new, 1)

    sub("| `src/report.py` | Builds downloadable PNG reports |",
        "| `src/report.py` | Builds downloadable PNG reports |\n"
        "| `src/camera_time.py` | California time for model inputs (the cameras' clock) |\n"
        "| `src/speed_limits.py` | Posted speed limit for a camera's road from OpenStreetMap |\n"
        "| `src/vehicle_tracking.py` | Vehicle tracking, camera calibration and speed estimation |\n"
        "| `src/live_monitor.py` | Background live monitor: detection, speed, automatic challans |\n"
        "| `src/live_video.py` | Plays a camera's live video stream in the page |\n"
        "| `src/route_map.py` | Road path, place search, cameras along a route |\n"
        "| `src/ui_theme.py` | The control-room look: header, badges, counter |\n"
        "| `cv_eval/` | Labelled frames and the vision evaluation (Lesson 11) |")
    sub("The project uses YOLOv8s.", "The project uses YOLOv8s with a detection confidence of 0.08, tuned on labelled frames (Lesson 11).")
    sub("| Tabs | single camera, monitoring grid, route |", "| Tabs | Single camera, Monitoring grid, Route, Live e-Challan |")
    sub("can download buffered frames faster than real time.",
        "can download buffered frames faster than real time.\n\n"
        "This is the older speed check in the Single camera tab. The newer Live e-Challan tab fits a camera model "
        "from the traffic itself (Lesson 12).")
    sub("The route tab lets users select multiple cameras. The final route congestion is\nthe worst segment.",
        "The route tab lets users pick cameras, search places, or click the map. It draws the real road path colored "
        "by nearby cameras (Lesson 15). The final route congestion is the worst segment.")
    sub("67 passed", "136 passed")
    sub("1. Historical dataset and live camera locations are not perfectly matched.",
        "1. The historical model is trained on Minneapolis-area data but the live cameras are in California.")
    sub("3. Speed estimation requires manual lane-width calibration.",
        "3. Speed estimates are not validated against radar (Lesson 12); the older speed check needs manual lane-width calibration.")
    sub("> The project is modular, with separate files",
        "> Beyond the prediction, a Live e-Challan tab runs detection and tracking on the live video, estimates speeds, and\n"
        "> generates clearly labelled simulated challans from real observed cars. I measured the vision part on 162 labelled\n"
        "> frames, which cut the counting error from 7.4 to 4.6 vehicles, and I documented what did not work.\n>\n"
        "> The project is modular, with separate files")
    sub("17. `tests/`", "17. `tests/`\n18. `src/vehicle_tracking.py`\n19. `src/live_monitor.py`\n20. `src/speed_limits.py`\n"
        "21. `src/camera_time.py`\n22. `src/route_map.py`\n23. `cv_eval/evaluate_cv.py`")
    return g


def split_h1(md: str):
    """(intro text before the first '##', rest) with the leading '# Title' removed."""
    lines = md.replace("\r\n", "\n").split("\n")
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    text = "\n".join(lines)
    k = text.find("\n## ")
    return (text[:k], text[k + 1:]) if k >= 0 else (text, "")


# ---------------------------------------------------------------- document
class Doc(BaseDocTemplate):
    def afterFlowable(self, fl):
        if isinstance(fl, Heading) and fl.toc_level is not None:
            self.notify("TOCEntry", (fl.toc_level, fl.toc_text, self.page))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "Smart Traffic Congestion Prediction  |  Demo and Study Guide")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build():
    demo = read(os.path.join(ROOT, "DEMO_SCRIPT.md"))
    guide = patch_guide(read(os.path.join(ROOT, "PROJECT_PRESENTATION_GUIDE.md")))
    new = read(os.path.join(HERE, "new_lessons.md"))
    qa_at = new.index("## Added Questions and Answers")
    new_lessons, new_qa = new[:qa_at], new[qa_at:]

    demo_intro, demo_rest = split_h1(demo)
    guide_intro, guide_rest = split_h1(guide)
    # new lessons go after Lesson 10 (before the 2-minute script); extra questions before the study order
    marker = "## Complete 2-Minute Presentation Script"
    guide_rest = guide_rest.replace(marker, new_lessons.rstrip() + "\n\n---\n\n" + marker, 1)
    guide_rest = guide_rest.replace("## Study Order", new_qa.rstrip() + "\n\n---\n\n## Study Order", 1)

    avail = A4[0] - 36 * mm
    doc = Doc(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
              title="Smart Traffic Congestion Prediction: Demo and Study Guide", author="Rakesh Tummala")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=footer)])

    toc = TableOfContents()
    toc.levelStyles = [S["toc1"], S["toc2"]]
    story = [
        Spacer(1, 40 * mm),
        Paragraph("Smart Traffic Congestion Prediction", S["cover_title"]),
        Paragraph("Live demo walkthrough and learning guide<br/>Computer vision + machine learning", S["cover_sub"]),
        Spacer(1, 6 * mm),
    ]
    how = Table([[Paragraph(
        "<b>How to use this document</b><br/><br/>"
        "<b>Part A: Live Demo Walkthrough.</b> Follow this while you present. It lists exactly what to click and say in the "
        "running app, with timings and fixes if something breaks.<br/><br/>"
        "<b>Part B: Learning Guide.</b> Read this before presenting so you can explain every part: the data, the models, "
        "the computer vision, the fusion, and the newer features (lessons 11 to 15), plus a 2-minute script and "
        "interview questions.<br/><br/>"
        "<b>Part C: Concepts and Talking Points.</b> Plain-language answers to \"what is X?\" and \"how does Y work?\": "
        "YOLO, CNN, OpenCV, OCR, how the three ML models are calculated and trained, the dataset, the decimal numbers "
        "and formulas, why California, what makes the project different, which cameras to use, and two scripts to "
        "read aloud.<br/><br/>"
        "<font color='#5B6B7A'>Prepared " + date.today().strftime("%d %B %Y") + ". The simulated challans are a demonstration, "
        "not real enforcement.</font>", S["body"])]], colWidths=[avail])
    how.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PANEL), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                             ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                             ("TOPPADDING", (0, 0), (-1, -1), 12), ("BOTTOMPADDING", (0, 0), (-1, -1), 12)]))
    story += [how, PageBreak(), Paragraph("Contents", S["h1"]), toc, PageBreak()]

    story.append(Heading("Part A: Live Demo Walkthrough", S["h1"], toc_level=0))
    story += flowables(parse(demo_intro), avail)
    story += flowables(parse(demo_rest), avail)

    story += [PageBreak(), Heading("Part B: Learning Guide", S["h1"], toc_level=0)]
    story += flowables(parse(guide_intro), avail)
    story += flowables(parse(guide_rest), avail, page_break_h2=False)

    concepts = read(os.path.join(HERE, "concepts.md"))
    concepts_intro, concepts_rest = split_h1(concepts)
    story += [PageBreak(), Heading("Part C: Concepts and Talking Points", S["h1"], toc_level=0)]
    story += flowables(parse(concepts_intro), avail)
    story += flowables(parse(concepts_rest), avail)

    doc.multiBuild(story)
    print("Wrote", OUT)


if __name__ == "__main__":
    build()
