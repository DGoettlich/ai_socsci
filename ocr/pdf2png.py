from pathlib import Path
import fitz  # PyMuPDF
from tqdm import tqdm

pdf_path = Path("data/ocr/newrecords18.pdf")
outdir = Path("data/ocr/pages")
outdir.mkdir(parents=True, exist_ok=True)

doc = fitz.open(pdf_path)

for i, page in tqdm(enumerate(doc), desc="converting", total=len(doc)):
    pix = page.get_pixmap(dpi=300)
    pix.save(outdir / f"page_{i + 1:03d}.png")

doc.close()
