"""
scripts/test_user_upload.py — Test Real User Upload Image against Live Backend API.
"""

import json
import requests
from pathlib import Path

API_URL = "http://localhost:8000/claims/process"

# Find any webp files in data/uploads/
uploads = list(Path("data/uploads").glob("*.webp")) + list(Path("data/uploads").glob("*medical*"))
print(f"Found specimen upload files: {uploads}")

for file_path in uploads:
    print(f"\nProcessing upload file: {file_path}")
    with open(file_path, "rb") as fh:
        files = {"file": (file_path.name, fh, "image/webp" if file_path.suffix == ".webp" else "image/png")}
        try:
            resp = requests.post(API_URL, files=files, timeout=180)
            resp.raise_for_status()
            res = resp.json()
            print("Extracted fields:")
            for k, v in res.get("extracted_fields", {}).items():
                val = v.get("value") if isinstance(v, dict) else str(v)
                status = v.get("status") if isinstance(v, dict) else "N/A"
                print(f"  {k:28s}: [{status:10s}] '{val}'")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
