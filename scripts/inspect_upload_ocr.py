"""
scripts/inspect_upload_ocr.py — Inspect raw OCR text of upload_eeffefc9.webp
"""

from pathlib import Path
import pytesseract
from PIL import Image as PILImage

img_path = Path("data/uploads/upload_eeffefc9.webp")
if img_path.exists():
    img = PILImage.open(img_path).convert("L")
    ocr_text = pytesseract.image_to_string(img)
    print("="*80)
    print(f"RAW OCR TEXT FOR {img_path}:")
    print("="*80)
    print(repr(ocr_text))
    print("\nPLAIN TEXT:")
    print(ocr_text)
else:
    print(f"File {img_path} not found.")
