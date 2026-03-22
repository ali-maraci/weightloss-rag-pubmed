import re
from lxml import etree


def clean_pmc_xml(raw_body_bytes: bytes) -> str:
    """
    Parse and clean PMC JATS XML into structured plain text suitable for chunking.

    Features:
    - Preserves section hierarchy using Markdown-style headers
    - Maintains paragraph boundaries
    - Removes references, tables, figures, formulas, supplementary material
    - Normalizes whitespace for readability
    """
    if not raw_body_bytes:
        return ""

    try:
        parser = etree.XMLParser(
            recover=True,
            remove_blank_text=True,
            resolve_entities=False
        )
        root = etree.fromstring(raw_body_bytes, parser=parser)
    except Exception:
        return ""

    body = root.find(".//body")
    if body is None:
        return ""

    # Remove unwanted elements
    for tag in body.xpath(
        ".//fig | .//table-wrap | .//table | .//ref-list | "
        ".//xref | .//sup | .//sub | .//disp-formula | "
        ".//inline-formula | .//media | .//supplementary-material"
    ):
        parent = tag.getparent()
        if parent is not None:
            parent.remove(tag)

    sections_text = [_extract_section(sec) for sec in body.findall("./sec")]

    cleaned_text = "\n\n".join(sections_text)
    return _normalize_blank_lines(cleaned_text)


def _extract_section(sec: etree.Element, level: int = 2) -> str:
    """
    Recursively extract section text from a <sec> element.
    Uses Markdown-style headers for hierarchy (#, ##, ###, etc.).
    Maintains paragraph boundaries for chunking.
    """
    section_parts = []

    title_el = sec.find("./title")
    if title_el is not None:
        title_text = _normalize_paragraph("".join(title_el.itertext()))
        if title_text:
            header = f"{'#' * min(level, 6)} {title_text}"
            section_parts.append(header)

    for p in sec.findall("./p"):
        paragraph_text = _normalize_paragraph("".join(p.itertext()))
        if paragraph_text:
            section_parts.append(paragraph_text)

    for child_sec in sec.findall("./sec"):
        section_parts.append(_extract_section(child_sec, level + 1))

    return "\n\n".join(section_parts)


def _normalize_paragraph(text: str) -> str:
    """Normalize whitespace inside a paragraph but preserve paragraph boundaries."""
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _normalize_blank_lines(text: str) -> str:
    """Normalize excessive blank lines in final output."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
