from pathlib import Path
import fitz  # PyMuPDF


pdf_path = Path("data/ocr/magazine.pdf")

doc = fitz.open(pdf_path)

for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=300)
    pix.save(f"data/ocr/magazine_page_{i + 1:03d}.png")

doc.close()
