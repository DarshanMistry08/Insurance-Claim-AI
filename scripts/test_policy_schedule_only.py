"""
scripts/test_policy_schedule_only.py — Isolated Policy Schedule Model Call Validation.

Sends Star Health Policy Schedule document to http://localhost:8000/claims/process
and prints the exact model call log output and extracted JSON values.
"""

import json
import requests
from pathlib import Path

API_URL = "http://localhost:8000/claims/process"
file_path = Path("data/uploads/test_star_health_policy_schedule.png")

print("================================================================================")
print("       TESTING POLICY_SCHEDULE EXTRACTION ENGINE & MODEL INVOCATION            ")
print("================================================================================")
print(f"Target Document File: {file_path}")

if not file_path.exists():
    print(f"ERROR: File {file_path} not found.")
    exit(1)

with open(file_path, "rb") as fh:
    files = {"file": (file_path.name, fh, "image/png")}
    try:
        resp = requests.post(API_URL, files=files, timeout=180)
        resp.raise_for_status()
        result_json = resp.json()
        
        print("\n" + "="*80)
        print("EXTRACTED FIELD RESULTS FOR POLICY SCHEDULE:")
        print("="*80)
        extracted = result_json.get("extracted_fields", {})
        for k, v in extracted.items():
            val = v.get("value") if isinstance(v, dict) else str(v)
            status = v.get("status") if isinstance(v, dict) else "N/A"
            method = v.get("extraction_method") if isinstance(v, dict) else "N/A"
            print(f"  {k:28s} : [{status:10s}] '{val}' (Method: {method})")

        print("\n" + "="*80)
        print("RAW JSON RESULT OUTPUT:")
        print("="*80)
        print(json.dumps(result_json, indent=2))

    except Exception as e:
        print(f"API Request Failed: {e}")
