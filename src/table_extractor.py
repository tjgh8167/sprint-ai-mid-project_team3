from __future__ import annotations

import io
from pathlib import Path


def extract_pdf_tables(path: str | Path) -> list[dict]:
    import fitz

    records = []
    document = fitz.open(path)
    try:
        for page_number, page in enumerate(document, start=1):
            for table_number, table in enumerate(page.find_tables().tables, start=1):
                markdown = table_rows_to_markdown(table.extract())
                if markdown:
                    records.append(
                        {
                            "page_number": page_number,
                            "table_number": table_number,
                            "table_markdown": markdown,
                        }
                    )
    finally:
        document.close()
    return records


def table_rows_to_markdown(rows: list[list[str | None]]) -> str:
    normalized_rows = [
        [str(cell or "").replace("|", r"\|").replace("\n", " ").strip() for cell in row]
        for row in rows
        if row
    ]
    if len(normalized_rows) < 2 or not normalized_rows[0]:
        return ""

    width = len(normalized_rows[0])
    normalized_rows = [
        row[:width] + [""] * max(0, width - len(row))
        for row in normalized_rows
    ]
    lines = [
        f"| {' | '.join(normalized_rows[0])} |",
        f"| {' | '.join(['---'] * width)} |",
    ]
    lines.extend(f"| {' | '.join(row)} |" for row in normalized_rows[1:])
    return "\n".join(lines)

def _count_line_runs(mask) -> int:
    import numpy as np

    padded = np.concatenate(([False], mask, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return int((ends - starts >= 2).sum())


def is_table_image(image_bytes: bytes) -> bool:
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            gray = np.asarray(image.convert("L"))
    except UnidentifiedImageError:
        return False

    dark_pixels = gray < 80
    horizontal_lines = _count_line_runs(dark_pixels.mean(axis=1) > 0.70)
    vertical_lines = _count_line_runs(dark_pixels.mean(axis=0) > 0.70)
    return horizontal_lines >= 2 and vertical_lines >= 2

def _line_centers(mask) -> list[int]:
    import numpy as np

    padded = np.concatenate(([False], mask, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [int((start + end - 1) / 2) for start, end in zip(starts, ends) if end - start >= 2]


def extract_image_table_markdown(
    image_bytes: bytes,
    language: str,
    *,
    page_seg_mode: int,
    image_scale: int,
    line_density: float,
    min_cell_width: int,
    min_cell_height: int,
) -> str:
    import numpy as np
    import pytesseract
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(image_bytes)) as image:
        prepared = ImageOps.autocontrast(image.convert("L"))
        pixels = np.asarray(prepared)
        dark_pixels = pixels < 80
        horizontal = _line_centers(dark_pixels.mean(axis=1) > line_density)
        vertical = _line_centers(dark_pixels.mean(axis=0) > line_density)
        if len(horizontal) < 3 or len(vertical) < 3:
            return ""

        rows = []
        for top, bottom in zip(horizontal, horizontal[1:]):
            row = []
            for left, right in zip(vertical, vertical[1:]):
                if right - left <= min_cell_width or bottom - top <= min_cell_height:
                    row.append("")
                    continue
                cell = prepared.crop((left + 1, top + 1, right, bottom))
                if image_scale > 1:
                    cell = cell.resize(
                        (cell.width * image_scale, cell.height * image_scale),
                        Image.Resampling.LANCZOS,
                    )
                row.append(
                    pytesseract.image_to_string(
                        cell,
                        lang=language,
                        config=f"--psm {page_seg_mode}",
                    ).strip().replace("\n", " ")
                )
            rows.append(row)
    return table_rows_to_markdown(rows)


def extract_table_ocr(
    image_bytes: bytes,
    language: str,
    page_seg_mode: int,
    image_scale: int,
) -> str:
    import pytesseract
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(image_bytes)) as image:
        prepared = ImageOps.autocontrast(image.convert("L"))
        if image_scale > 1:
            prepared = prepared.resize(
                (prepared.width * image_scale, prepared.height * image_scale),
                Image.Resampling.LANCZOS,
            )
        return pytesseract.image_to_string(
            prepared,
            lang=language,
            config=f"--psm {page_seg_mode}",
        ).strip()
