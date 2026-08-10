"""
scripts/test_fix_upload.py — Test noise-tolerant OCR LLM extraction on upload_eeffefc9.webp
"""

import os
import re
import json
import requests
from pathlib import Path

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

import pytesseract
from PIL import Image as PILImage

img_path = Path("data/uploads/upload_eeffefc9.webp")
img = PILImage.open(img_path).convert("L")
ocr_text = pytesseract.image_to_string(img)

prompt = f"""You are an expert insurance document reader.
Your task is to extract exact field values from noisy OCR text of an insurance Policy Schedule document.

OCR Text of Document:
\"\"\"{ocr_text}\"\"\"

Target Fields to Extract for policy_schedule:
1. policy_number (e.g. P/700002/01/2023/007530 or similar policy number)
2. previous_policy_number (previous or prior policy number if present)
3. proposer_name (name of proposer / insured person e.g. ANANT KUMAR)
4. proposer_address (full address of proposer e.g. 3A, Kunwar Singh Chowk...)
5. issuing_office_address (issuing office or branch address)
6. sum_insured (sum insured or coverage limit e.g. Rs 3,00,000)
7. premium (total or gross premium e.g. Rs 29,973 or Rs 25,000)
8. period_of_insurance_from (policy start date e.g. 03/07/2022)
9. period_of_insurance_to (policy expiry date e.g. 02/07/2023)
10. scheme_description (plan description e.g. Basic Floater / Family Optima)
11. insured_persons (names of covered family members)
12. intermediary_name (agent/broker/intermediary name e.g. Delhi Telesales)
13. policy_type (e.g. Health Insurance)

Return ONLY a JSON object mapping each key to:
{{
  "value": "extracted text value or empty if missing",
  "confidence": 0.95,
  "status": "EXTRACTED" or "NOT_FOUND"
}}
"""

print(f"Calling Ollama model '{OLLAMA_MODEL}' with OCR text prompt (no image payload)...")
resp = requests.post(
    f"{OLLAMA_BASE_URL}/api/generate",
    json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
    timeout=60
)
resp.raise_for_status()
raw = resp.json().get("response", "").strip()

print("\n" + "="*80)
print("RAW OLLAMA LLM RESPONSE ON NOISY OCR:")
print("="*80)
print(raw)
