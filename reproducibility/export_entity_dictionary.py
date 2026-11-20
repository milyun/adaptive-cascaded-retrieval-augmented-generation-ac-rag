#!/usr/bin/env python3
"""Export the 80 generic entity terms from the source Appendix 4 DOCX."""

import argparse
import csv
import re
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree


WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
EXPECTED_TERMS = 80


def extract_cell(cell: ElementTree.Element) -> str:
    return "".join(node.text or "" for node in cell.findall(".//w:t", WORD_NS)).strip()


def read_dictionary(document: Path) -> list:
    with ZipFile(document) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    tables = root.findall(".//w:tbl", WORD_NS)
    if len(tables) != 1:
        raise ValueError("Expected one glossary table in Appendix 4")
    rows = []
    category = None
    for row in tables[0].findall("./w:tr", WORD_NS)[1:]:
        cells = [extract_cell(cell) for cell in row.findall("./w:tc", WORD_NS)]
        if len(cells) != 5:
            raise ValueError("Unexpected Appendix 4 table schema")
        if cells[0]:
            category = re.sub(r"\s*\(\d+\)\s*$", "", cells[0])
        if not category or not cells[2] or not cells[3]:
            raise ValueError("Incomplete glossary row")
        rows.append({
            "category": category,
            "term": cells[2],
            "english_gloss": cells[3],
            "generic_term": "true" if cells[4].startswith("✓") else "false",
        })
    if len(rows) != EXPECTED_TERMS:
        raise ValueError(f"Expected {EXPECTED_TERMS} terms, found {len(rows)}")
    if len({(row["category"], row["term"]) for row in rows}) != EXPECTED_TERMS:
        raise ValueError("Glossary terms must be unique within category")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appendix-docx", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "entity_recall_audit" / "entity_dictionary.csv")
    args = parser.parse_args()
    rows = read_dictionary(args.appendix_docx)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["category", "term", "english_gloss", "generic_term"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} de-identified generic terms to {args.output}")


if __name__ == "__main__":
    main()
