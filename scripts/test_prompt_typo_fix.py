"""
scripts/test_prompt_typo_fix.py — Test OCR Typo Tolerant Prompt on upload_eeffefc9.webp
"""

import os
import json
import requests
from pathlib import Path
import pytesseract
from PIL import Image as PILImage

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

img_path = Path("data/uploads/upload_eeffefc9.webp")
img = PILImage.open(img_path).convert("L")
ocr_text = pytesseract.image_to_string(img)

prompt = f"""You are an expert insurance document reader specializing in noisy/scanned OCR extraction.
CLASSIFIED DOCUMENT TYPE: POLICY_SCHEDULE

CRITICAL EXTRACTION RULES:
1. ONLY extract fields belonging to the policy_schedule target schema.
2. OCR TYPO TOLERANT EXTRACTION: Extract the true field values even if OCR output has minor spelling typos in labels or values (e.g., 'Proporefs Noses' means 'Proposer Name', 'PERIOD OF WSURANCE' means 'Period of Insurance', 'LIMIT OF COVERAGE' or 'SUMINSURED' means 'Sum Insured').
3. Clean up obvious OCR noise from values (e.g. if policy number is ']_PITaOOOz OT 2O7STOOTEIO', extract the actual policy number or formatted string).

Schema Fields to Extract:
- policy_number: Policy schedule or policy number identifier
- previous_policy_number: Previous/prior policy number if present
- proposer_name: Full name of policyholder / proposer (e.g. ANANT KUMAR)
- proposer_address: Full address of proposer (e.g. 3A, Kunwar Singh Chowk...)
- sum_insured: Coverage limit or sum insured amount (e.g. Rs 3,00,000)
- premium: Gross or total premium amount (e.g. Rs 29,973 or Rs 25,000)
- period_of_insurance_from: Policy start date (e.g. 03/07/2022)
- period_of_insurance_to: Policy expiry date (e.g. 02/07/2023)
- scheme_description: Plan description (e.g. Basic Floater / Family Health Optima)
- insured_persons: Covered family member names
- intermediary_name: Agent or intermediary name (e.g. Delhi Telesales)
- policy_type: Type of policy (e.g. Health Insurance)

Raw OCR Text:
\"\"\"{ocr_text}\"\"\"

Return ONLY a JSON object mapping each field to:
{{
  "value": "extracted value or empty if missing",
  "status": "EXTRACTED" or "NOT_FOUND"
}}
"""

print(f"Calling Ollama model '{OLLAMA_MODEL}' with OCR typo-tolerant prompt...")
resp = requests.post(
    f"{OLLAMA_BASE_URL}/api/generate",
    json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
    timeout=60
)
resp.raise_for_status()
raw = resp.json().get("response", "").strip()

print("\n" + "="*80)
print("EXTRACTED RESULT FOR upload_eeffefc9.webp:")
print("="*80)
print(raw)
