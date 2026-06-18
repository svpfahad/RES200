"""Accept all tracked changes and strip all comments from the JURI editor's docx.

Uses lxml to walk the XML tree (regex was unreliable with self-closing
<w:del .../> markers).
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

from lxml import etree

ROOT = Path(r"C:\Users\Fahad\Downloads\RES200")
SRC = ROOT / "JURI-00032-2025" / "KFUPM_309_JURI-00032-2025.docx"
OUT = ROOT / "_build_body.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{W_NS}}}"

COMMENT_PARTS = {
    "word/comments.xml",
    "word/commentsIds.xml",
    "word/commentsExtended.xml",
    "word/commentsExtensible.xml",
}

# Tags whose entire subtree should be removed (deletion markers and revision metadata)
KILL_TAGS = {
    f"{W}del",          # tracked deletion (block) AND deletion marker (self-closing)
    f"{W}rPrChange",    # revision metadata for run properties
    f"{W}pPrChange",    # revision metadata for paragraph properties
    f"{W}sectPrChange",
    f"{W}tblPrChange",
    f"{W}trPrChange",
    f"{W}tcPrChange",
    f"{W}numberingChange",
    f"{W}commentRangeStart",
    f"{W}commentRangeEnd",
    f"{W}commentReference",
}

# Tags that should be unwrapped (children promoted to parent's children)
UNWRAP_TAGS = {
    f"{W}ins",          # tracked insertion (block) — keep the inner runs
    f"{W}moveTo",       # accepted move target — keep contents
}

# Tags removed if self-closing (marker form) — should never appear after deletion above,
# but keep as defensive belt.
KILL_MARKERS = {
    f"{W}moveFrom",
    f"{W}moveFromRangeStart",
    f"{W}moveFromRangeEnd",
    f"{W}moveToRangeStart",
    f"{W}moveToRangeEnd",
}


def kill_subtree(elem: etree._Element) -> None:
    parent = elem.getparent()
    if parent is None:
        return
    parent.remove(elem)


def unwrap(elem: etree._Element) -> None:
    """Replace `elem` with its children, preserving text/tail."""
    parent = elem.getparent()
    if parent is None:
        return
    idx = list(parent).index(elem)
    # Build the list of children and move them up
    children = list(elem)
    # Concatenate elem.text into the previous sibling's tail or parent.text
    head_text = elem.text or ""
    if head_text:
        if idx == 0:
            parent.text = (parent.text or "") + head_text
        else:
            prev = parent[idx - 1]
            prev.tail = (prev.tail or "") + head_text
    # Insert children where elem was, in order
    for offset, child in enumerate(children):
        parent.insert(idx + offset, child)
    # Append elem.tail to the last moved child or to whatever was before
    tail = elem.tail or ""
    if children and tail:
        last = children[-1]
        last.tail = (last.tail or "") + tail
    elif tail:
        # No children — push tail back into parent's tail context
        if idx == 0:
            parent.text = (parent.text or "") + tail
        else:
            prev = parent[idx - 1]
            prev.tail = (prev.tail or "") + tail
    # Now remove the empty wrapper
    parent.remove(elem)


def accept_changes(xml_bytes: bytes) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(xml_bytes, parser)

    # Walk a snapshot of all elements (we'll mutate the tree)
    for elem in list(root.iter()):
        if elem.getparent() is None:
            continue
        tag = elem.tag
        if tag in KILL_TAGS or tag in KILL_MARKERS:
            kill_subtree(elem)
        elif tag in UNWRAP_TAGS:
            unwrap(elem)

    return etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True
    )


def strip_track_changes_setting(xml_bytes: bytes) -> bytes:
    root = etree.fromstring(xml_bytes)
    for elem in list(root.iter(f"{W}trackChanges")):
        kill_subtree(elem)
    return etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True
    )


def strip_comment_relationships(xml_bytes: bytes) -> bytes:
    PKG_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
    root = etree.fromstring(xml_bytes)
    for rel in list(root):
        rtype = rel.get("Type", "")
        if "comments" in rtype:
            root.remove(rel)
    return etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True
    )


def strip_comment_content_types(xml_bytes: bytes) -> bytes:
    CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
    root = etree.fromstring(xml_bytes)
    for child in list(root):
        if child.tag.endswith("Override"):
            partname = child.get("PartName", "")
            if "/word/comments" in partname:
                root.remove(child)
    return etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True
    )


def main() -> None:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found", file=sys.stderr)
        sys.exit(1)
    if OUT.exists():
        OUT.unlink()

    with zipfile.ZipFile(SRC, "r") as src, zipfile.ZipFile(
        OUT, "w", zipfile.ZIP_DEFLATED
    ) as dst:
        for item in src.infolist():
            name = item.filename
            if name in COMMENT_PARTS:
                continue
            data = src.read(name)
            if name == "word/document.xml":
                data = accept_changes(data)
            elif name == "word/settings.xml":
                data = strip_track_changes_setting(data)
            elif name == "word/_rels/document.xml.rels":
                data = strip_comment_relationships(data)
            elif name == "[Content_Types].xml":
                data = strip_comment_content_types(data)
            dst.writestr(item, data)

    # Strict verification using lxml
    with zipfile.ZipFile(OUT) as z:
        body = z.read("word/document.xml")
    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(body, parser)
    if parser.error_log:
        print("XML errors:")
        for err in parser.error_log:
            print(f"  L{err.line}:C{err.column} - {err.message}")
        sys.exit(2)

    leftovers = []
    for tag in KILL_TAGS | UNWRAP_TAGS | KILL_MARKERS:
        count = len(list(root.iter(tag)))
        if count:
            leftovers.append((tag, count))
    if leftovers:
        print("LEFTOVER tracked-change elements:")
        for t, c in leftovers:
            print(f"  {t}: {c}")
        sys.exit(3)
    print("OK: tracked changes accepted, comments stripped, XML valid")
    print(f"Body docx written to {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
