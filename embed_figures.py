"""Replace word/media/image1..7.png inside the cleaned docx with the regenerated PNGs."""
import zipfile
from pathlib import Path

ROOT = Path(r"C:\Users\Fahad\Downloads\RES200")
SRC = ROOT / "_build_body.docx"
NEW_FIGS = ROOT / "figs_revised"
OUT = ROOT / "_build_with_figs.docx"

if OUT.exists():
    OUT.unlink()

# Names we will overwrite
TARGETS = {f"word/media/image{i}.png" for i in range(1, 8)}

with zipfile.ZipFile(SRC, "r") as src, zipfile.ZipFile(
    OUT, "w", zipfile.ZIP_DEFLATED
) as dst:
    for item in src.infolist():
        name = item.filename
        if name in TARGETS:
            num = name.rsplit("image", 1)[1].split(".")[0]
            new_path = NEW_FIGS / f"image{num}.png"
            data = new_path.read_bytes()
            dst.writestr(name, data)
        else:
            dst.writestr(item, src.read(name))

print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
print("Embedded figure sizes:")
with zipfile.ZipFile(OUT) as z:
    for n in sorted(z.namelist()):
        if "media/image" in n:
            print(f"  {n}: {z.getinfo(n).file_size:,} bytes")
