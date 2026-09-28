"""Tests for CV loading and redaction."""

import zipfile
from pathlib import Path

import pytest

from jobagent.cv import load_cv, redact


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail me: jane.doe+cv@example.ie", "mail me: [email]"),
        ("+353 85 222 2098 | Dublin", "[phone] | Dublin"),
        ("call (01) 234 5678", "call [phone]"),
        ("085-222-2098", "[phone]"),
    ],
)
def test_redact_removes_contact_details(text: str, expected: str) -> None:
    assert redact(text) == expected


@pytest.mark.parametrize(
    "text", ["MSc AI | 2025 - 2026", "Hit@4 from 86% to 98%", "March 2023 - September 2023"]
)
def test_redact_keeps_dates_and_metrics(text: str) -> None:
    assert redact(text) == text


def _write_docx(path: Path, paragraphs: list[str]) -> None:
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    xml = f'<w:document xmlns:w="w"><w:body>{body}<w:p></w:p></w:body></w:document>'
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", xml)


def test_load_docx_extracts_paragraphs_and_redacts(tmp_path: Path) -> None:
    cv = tmp_path / "cv.docx"
    _write_docx(cv, ["Jane Doe", "jane@example.ie", "Python &amp; SQL &lt;3"])
    assert load_cv(cv) == "Jane Doe\n[email]\nPython & SQL <3"


def test_load_text_formats(tmp_path: Path) -> None:
    cv = tmp_path / "cv.md"
    cv.write_text("# Skills\nPython", encoding="utf-8")
    assert load_cv(str(cv)) == "# Skills\nPython"


def test_load_rejects_unsupported_and_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        load_cv(tmp_path / "cv.pdf")
    empty = tmp_path / "cv.txt"
    empty.write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="no text"):
        load_cv(empty)
