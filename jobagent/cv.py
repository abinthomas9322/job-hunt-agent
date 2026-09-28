"""Load a CV as plain text, with contact details removed.

Supports .docx (read straight from the Word XML, no extra dependency) and
plain-text formats (.txt, .md). Emails and phone numbers are redacted before
the text goes anywhere near an LLM API: the model needs your skills, not your
contact details.
"""

import re
import zipfile
from pathlib import Path

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# International or local phone numbers: +353 85 222 2098, (01) 234 5678, 085-222-2098
PHONE = re.compile(r"(?<!\w)(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?){2,4}\d{3,4}(?!\w)")


def redact(text: str) -> str:
    """Replace email addresses and phone numbers with placeholders."""
    return PHONE.sub("[phone]", EMAIL.sub("[email]", text))


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8")
    paragraphs = re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)
    lines = ("".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p)).strip() for p in paragraphs)
    text = "\n".join(line for line in lines if line)
    return (
        text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    )


def load_cv(path: str | Path) -> str:
    """Read a CV file and return its redacted plain text.

    Raises:
        ValueError: For unsupported file types or an empty CV.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".docx":
        text = _docx_text(p)
    elif suffix in {".txt", ".md"}:
        text = p.read_text(encoding="utf-8")
    else:
        raise ValueError(f"unsupported CV format {suffix!r}: use .docx, .txt or .md")
    if not text.strip():
        raise ValueError("the CV file contains no text")
    return redact(text)
