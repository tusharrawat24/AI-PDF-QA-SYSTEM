import re
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def convert_inline_markdown(text: str) -> str:
    """
    Convert simple Markdown formatting into ReportLab-compatible markup.
    """

    text = escape(text.strip())

    # Links: [label](url) -> label only
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text,
    )

    # Inline code (before bold/italic so markers inside code stay intact)
    text = re.sub(
        r"`(.+?)`",
        r"<font name='Courier'>\1</font>",
        text,
    )

    # Bold text
    text = re.sub(
        r"\*\*(.+?)\*\*",
        r"<b>\1</b>",
        text,
    )

    # Italic text with single asterisks
    text = re.sub(
        r"(?<!\*)\*([^*]+)\*(?!\*)",
        r"<i>\1</i>",
        text,
    )

    # Italic text with underscores (word-boundary safe,
    # so snake_case or file names are not mangled)
    text = re.sub(
        r"(?<!\w)_([^_]+)_(?!\w)",
        r"<i>\1</i>",
        text,
    )

    # Strikethrough
    text = re.sub(
        r"~~(.+?)~~",
        r"<strike>\1</strike>",
        text,
    )

    return text


def is_markdown_table_separator(line: str) -> bool:
    """
    Check whether a Markdown table row is the separator row.
    """

    cells = [
        cell.strip()
        for cell in line.strip().strip("|").split("|")
    ]

    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", cell) is not None
        for cell in cells
    )


def parse_markdown_table(lines: list[str]) -> list[list[str]]:
    """
    Convert Markdown table lines into a list of table rows.
    """

    rows = []

    for line in lines:
        if is_markdown_table_separator(line):
            continue

        cells = [
            convert_inline_markdown(cell)
            for cell in line.strip().strip("|").split("|")
        ]

        rows.append(cells)

    return rows


def create_pdf(title: str, content: str) -> bytes:
    """
    Create a professionally formatted PDF from AI-generated Markdown text.
    """

    if not content or not content.strip():
        raise ValueError("The PDF content cannot be empty.")

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title=title,
        author="AI PDF Assistant",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name="DocumentTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=26,
        alignment=TA_CENTER,
        spaceAfter=20,
    )

    heading_1_style = ParagraphStyle(
        name="HeadingLevel1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        spaceBefore=16,
        spaceAfter=9,
        textColor=colors.HexColor("#1F2937"),
    )

    heading_2_style = ParagraphStyle(
        name="HeadingLevel2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=13,
        spaceAfter=7,
        textColor=colors.HexColor("#253858"),
    )

    heading_3_style = ParagraphStyle(
        name="HeadingLevel3",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor("#374151"),
    )

    body_style = ParagraphStyle(
        name="DocumentBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=16,
        spaceAfter=7,
        textColor=colors.HexColor("#111827"),
    )

    bullet_style = ParagraphStyle(
        name="DocumentBullet",
        parent=body_style,
        leftIndent=5,
        firstLineIndent=0,
        spaceAfter=3,
    )

    numbered_style = ParagraphStyle(
        name="DocumentNumbered",
        parent=body_style,
        leftIndent=10,
        firstLineIndent=-10,
        spaceAfter=6,
    )

    table_cell_style = ParagraphStyle(
        name="TableCell",
        parent=body_style,
        fontSize=8.5,
        leading=11,
        spaceAfter=0,
    )

    story = [
        Paragraph(escape(title), title_style),
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor("#CBD5E1"),
            spaceAfter=14,
        ),
    ]

    lines = content.splitlines()
    index = 0
    pending_bullets = []

    def flush_bullets() -> None:
        nonlocal pending_bullets

        if pending_bullets:
            story.append(
                ListFlowable(
                    pending_bullets,
                    bulletType="bullet",
                    start="circle",
                    leftIndent=20,
                    bulletFontName="Helvetica",
                    bulletFontSize=7,
                    spaceAfter=8,
                )
            )
            pending_bullets = []

    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.strip()

        if not line:
            flush_bullets()
            story.append(Spacer(1, 5))
            index += 1
            continue

        # Horizontal separator
        if line in {"---", "***", "___"}:
            flush_bullets()

            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.7,
                    color=colors.HexColor("#D1D5DB"),
                    spaceBefore=6,
                    spaceAfter=9,
                )
            )

            index += 1
            continue

        # Markdown table
        if "|" in line:
            table_lines = [line]
            next_index = index + 1

            while (
                next_index < len(lines)
                and "|" in lines[next_index]
                and lines[next_index].strip()
            ):
                table_lines.append(lines[next_index].strip())
                next_index += 1

            if (
                len(table_lines) >= 2
                and is_markdown_table_separator(table_lines[1])
            ):
                flush_bullets()

                parsed_rows = parse_markdown_table(table_lines)

                if parsed_rows:
                    table_data = []

                    for row_number, row in enumerate(parsed_rows):
                        table_row = [
                            Paragraph(
                                cell,
                                table_cell_style,
                            )
                            for cell in row
                        ]

                        table_data.append(table_row)

                    column_count = max(
                        len(row)
                        for row in parsed_rows
                    )

                    available_width = A4[0] - (
                        document.leftMargin
                        + document.rightMargin
                    )

                    column_widths = [
                        available_width / column_count
                    ] * column_count

                    table = Table(
                        table_data,
                        colWidths=column_widths,
                        repeatRows=1,
                        hAlign="LEFT",
                    )

                    table.setStyle(
                        TableStyle(
                            [
                                (
                                    "BACKGROUND",
                                    (0, 0),
                                    (-1, 0),
                                    colors.HexColor("#E2E8F0"),
                                ),
                                (
                                    "TEXTCOLOR",
                                    (0, 0),
                                    (-1, 0),
                                    colors.HexColor("#111827"),
                                ),
                                (
                                    "FONTNAME",
                                    (0, 0),
                                    (-1, 0),
                                    "Helvetica-Bold",
                                ),
                                (
                                    "VALIGN",
                                    (0, 0),
                                    (-1, -1),
                                    "TOP",
                                ),
                                (
                                    "GRID",
                                    (0, 0),
                                    (-1, -1),
                                    0.5,
                                    colors.HexColor("#CBD5E1"),
                                ),
                                (
                                    "ROWBACKGROUNDS",
                                    (0, 1),
                                    (-1, -1),
                                    [
                                        colors.white,
                                        colors.HexColor("#F8FAFC"),
                                    ],
                                ),
                                (
                                    "LEFTPADDING",
                                    (0, 0),
                                    (-1, -1),
                                    6,
                                ),
                                (
                                    "RIGHTPADDING",
                                    (0, 0),
                                    (-1, -1),
                                    6,
                                ),
                                (
                                    "TOPPADDING",
                                    (0, 0),
                                    (-1, -1),
                                    6,
                                ),
                                (
                                    "BOTTOMPADDING",
                                    (0, 0),
                                    (-1, -1),
                                    6,
                                ),
                            ]
                        )
                    )

                    story.append(table)
                    story.append(Spacer(1, 10))

                index = next_index
                continue

        # Headings (deeper Markdown levels collapse onto the
        # three available visual heading styles)
        if line.startswith("####### "):
            flush_bullets()

            story.append(
                Paragraph(
                    convert_inline_markdown(line[8:]),
                    heading_3_style,
                )
            )

        elif line.startswith("###### "):
            flush_bullets()

            story.append(
                Paragraph(
                    convert_inline_markdown(line[7:]),
                    heading_3_style,
                )
            )

        elif line.startswith("##### "):
            flush_bullets()

            story.append(
                Paragraph(
                    convert_inline_markdown(line[6:]),
                    heading_3_style,
                )
            )

        elif line.startswith("#### "):
            flush_bullets()

            story.append(
                Paragraph(
                    convert_inline_markdown(line[5:]),
                    heading_3_style,
                )
            )

        elif line.startswith("### "):
            flush_bullets()

            story.append(
                Paragraph(
                    convert_inline_markdown(line[4:]),
                    heading_3_style,
                )
            )

        elif line.startswith("## "):
            flush_bullets()

            story.append(
                Paragraph(
                    convert_inline_markdown(line[3:]),
                    heading_2_style,
                )
            )

        elif line.startswith("# "):
            flush_bullets()

            story.append(
                Paragraph(
                    convert_inline_markdown(line[2:]),
                    heading_1_style,
                )
            )

        # Bullets
        elif line.startswith(("- ", "* ", "• ")):
            bullet_text = line[2:].strip()

            pending_bullets.append(
                ListItem(
                    Paragraph(
                        convert_inline_markdown(bullet_text),
                        bullet_style,
                    )
                )
            )

        # Numbered list
        elif re.match(r"^\d+[\.\)]\s+", line):
            flush_bullets()

            story.append(
                Paragraph(
                    convert_inline_markdown(line),
                    numbered_style,
                )
            )

        # Normal paragraph
        else:
            flush_bullets()

            story.append(
                Paragraph(
                    convert_inline_markdown(line),
                    body_style,
                )
            )

        index += 1

    flush_bullets()

    document.build(story)

    pdf_data = buffer.getvalue()
    buffer.close()

    return pdf_data
