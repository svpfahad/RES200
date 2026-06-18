"""Apply front-matter edits to _build_with_figs.docx and write the final docx.

Edits:
1. Remove the corresponding-author '*' marker after 'Fahad Atwi'.
2. Remove the stray apostrophe ' (subscript) sitting between Mohammed Al-Khater's
   footnote markers — leftover formatting from the editor's pass.
3. Keep the trailing '*' on Mohammed Al-Khater (he is the corresponding author).
4. Insert a new paragraph below the author line with ORCID-iD placeholders for
   Fahad and the mentor — to be filled in before submission.
"""
from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

from lxml import etree

ROOT = Path(r"C:\Users\Fahad\Downloads\RES200")
SRC = ROOT / "_build_with_figs.docx"
OUT = ROOT / "XGBoost_pKa_Prediction_REVISION_FINAL.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{W_NS}}}"


def edit_document_xml(xml_bytes: bytes) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(xml_bytes, parser)

    # ---------------------------------------------------------------- 1 & 2
    # Walk the body looking for the author paragraph (the one containing
    # 'Fahad Atwi'), then strip:
    #   - the run containing <w:t>*</w:t> that appears between Fahad and Moayad,
    #     plus its bookmark wrappers;
    #   - the stray run containing <w:t>’</w:t> with subscript vertAlign that
    #     appears between Mohammed Al-Khater's footnote markers.
    paragraphs = root.iter(f"{W}p")
    author_p = None
    for p in paragraphs:
        text = "".join(t.text or "" for t in p.iter(f"{W}t"))
        if "Fahad Atwi" in text and "Mohammed Al-Khater" in text:
            author_p = p
            break

    if author_p is None:
        raise RuntimeError("Could not find author paragraph")

    # --- Remove the * after Fahad ---
    # Find the bookmark with name '_Hlk198588173' and remove from
    # <w:bookmarkStart .../> through <w:bookmarkEnd ... w:id="X"/> inclusive,
    # along with any run between them.
    bm_start = None
    for el in author_p.iter(f"{W}bookmarkStart"):
        if el.get(f"{W}name") == "_Hlk198588173":
            bm_start = el
            break

    if bm_start is not None:
        target_id = bm_start.get(f"{W}id")
        # Walk forward through siblings of bm_start until we find bookmarkEnd id=target_id
        parent = bm_start.getparent()
        children = list(parent)
        i_start = children.index(bm_start)
        i_end = None
        for j in range(i_start + 1, len(children)):
            ch = children[j]
            if ch.tag == f"{W}bookmarkEnd" and ch.get(f"{W}id") == target_id:
                i_end = j
                break
        if i_end is None:
            i_end = i_start  # fallback — just remove the start
        # Remove children in [i_start .. i_end] inclusive
        for ch in children[i_start : i_end + 1]:
            parent.remove(ch)

    # --- Remove the stray subscript ’ run between footnotes 4 and 5 ---
    runs_to_remove = []
    for r in author_p.iter(f"{W}r"):
        # A run with vertAlign=subscript and text '’'
        rPr = r.find(f"{W}rPr")
        if rPr is None:
            continue
        va = rPr.find(f"{W}vertAlign")
        if va is None or va.get(f"{W}val") != "subscript":
            continue
        text = "".join((t.text or "") for t in r.iter(f"{W}t"))
        if text.strip() in {"’", "'"}:
            runs_to_remove.append(r)
    for r in runs_to_remove:
        r.getparent().remove(r)

    # ---------------------------------------------------------------- 3
    # Insert ORCID placeholder paragraph immediately after the author paragraph.
    # We replicate the existing paragraph's formatting by cloning the empty
    # paragraph that sits below the author block (it shares the same style).
    #
    # XML for the new paragraph:
    #   <w:p>
    #     <w:pPr><w:spacing w:after="200"/></w:pPr>
    #     <w:r><w:rPr><w:rFonts ascii=Arial.../><w:i/><w:szCs val=24/></w:rPr>
    #       <w:t>ORCID iDs: Fahad Atwi …; Mohammed Al-Khater …</w:t>
    #     </w:r>
    #   </w:p>

    parent = author_p.getparent()
    idx = list(parent).index(author_p)

    nsmap = {"w": W_NS}
    orcid_xml = """
        <w:p xmlns:w="{ns}">
          <w:pPr>
            <w:spacing w:after="200"/>
            <w:jc w:val="center"/>
            <w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/><w:i/><w:szCs w:val="24"/></w:rPr>
          </w:pPr>
          <w:r>
            <w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/><w:i/><w:szCs w:val="24"/></w:rPr>
            <w:t xml:space="preserve">ORCID: Fahad Atwi [paste iD here]; Mohammed Al-Khater [paste mentor iD here]</w:t>
          </w:r>
        </w:p>
    """.format(ns=W_NS).strip()
    orcid_p = etree.fromstring(orcid_xml)
    parent.insert(idx + 1, orcid_p)

    return etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True
    )


def main() -> None:
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(SRC, "r") as src, zipfile.ZipFile(
        OUT, "w", zipfile.ZIP_DEFLATED
    ) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "word/document.xml":
                data = edit_document_xml(data)
            dst.writestr(item, data)

    # Verify by reading the author paragraph back
    with zipfile.ZipFile(OUT) as z:
        body = z.read("word/document.xml").decode("utf-8")
    # Quick text rendering of paragraphs
    from docx import Document
    doc = Document(OUT)
    for i, p in enumerate(doc.paragraphs[:6]):
        print(f"P{i} [{p.style.name}]: {p.text}")
    print(f"\nFinal docx at: {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
