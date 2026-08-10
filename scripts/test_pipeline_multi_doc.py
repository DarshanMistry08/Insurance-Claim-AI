"""
scripts/test_pipeline_multi_doc.py — STEP 8 Mandatory Cross-Type Regression Testing Matrix.

Executes cross-type regression test matrix against all 4 fixture document types:
  1. Star Health Policy Schedule (Star Health Family Optima with Patna fields)
  2. Real Claim Form (with genuine incident data)
  3. Medical Bill Invoice
  4. KYC ID Proof (PAN Card)

Reports field-by-field pass/fail status and prints actual extracted JSON and summary text.
"""

import json
import requests
from pathlib import Path

API_URL = "http://localhost:8000/claims/process"
UPLOADS_DIR = Path("data/uploads")

test_files = [
    ("policy_schedule", "Star Health Policy Schedule", UPLOADS_DIR / "test_star_health_policy_schedule.png"),
    ("claim_form", "Claim Form Notification", UPLOADS_DIR / "test_claim_form.png"),
    ("medical_bill_invoice", "Medical Bill Invoice", UPLOADS_DIR / "test_medical_bill.png"),
    ("kyc_id_proof", "KYC ID Proof (PAN Card)", UPLOADS_DIR / "test_kyc_id_proof.png"),
]

print("================================================================================")
print("             STEP 8: MANDATORY CROSS-TYPE REGRESSION TESTING MATRIX             ")
print("================================================================================\n")

for doc_type, label, file_path in test_files:
    print("\n" + "="*80)
    print(f"FIXTURE TEST: {label} (Expected Type: {doc_type})")
    print(f"File Path: {file_path}")
    print("="*80)

    if not file_path.exists():
        print(f"ERROR: File {file_path} not found.")
        continue

    with open(file_path, "rb") as fh:
        files = {"file": (file_path.name, fh, "image/png")}
        try:
            resp = requests.post(API_URL, files=files, timeout=180)
            resp.raise_for_status()
            result_json = resp.json()
            
            classified_type = result_json.get("document_type", "unknown")
            conf = result_json.get("document_type_confidence", 0.0)
            
            print(f"\n[CLASSIFICATION]: Classified as '{classified_type}' (Confidence: {conf:.0%})")
            print(f"[CLASSIFICATION MATCH]: {'PASS' if classified_type == doc_type else 'FAIL'}")
            
            print("\n--- EXTRACTED FIELDS ---")
            extracted = result_json.get("extracted_fields", {})
            for k, v in extracted.items():
                val = v.get("value") if isinstance(v, dict) else str(v)
                status = v.get("status") if isinstance(v, dict) else "N/A"
                print(f"  {k:28s}: [{status:10s}] '{val}'")

            print("\n--- SUMMARY TEXT ---")
            summary = result_json.get("final_summary", {})
            summary_text = summary.get("summary_text") if isinstance(summary, dict) else str(summary)
            print(summary_text)

            print("\n--- RAW JSON RESULT OUTPUT ---")
            print(json.dumps(result_json, indent=2))

        except Exception as e:
            print(f"API Request Failed for {label}: {e}")
