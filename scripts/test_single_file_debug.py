"""
scripts/test_single_file_debug.py — Single File Direct Debug on upload_eeffefc9.webp
"""

import json
import requests
from pathlib import Path

API_URL = "http://localhost:8000/claims/process"
file_path = Path("data/uploads/upload_eeffefc9.webp")

print(f"Testing direct API upload for: {file_path}")
with open(file_path, "rb") as fh:
    files = {"file": (file_path.name, fh, "image/webp")}
    resp = requests.post(API_URL, files=files, timeout=120)
    print("STATUS CODE:", resp.status_code)
    res = resp.json()
    print("DOCUMENT TYPE:", res.get("document_type"), res.get("document_type_confidence"))
    print("\nEXTRACTED FIELDS:")
    print(json.dumps(res.get("extracted_fields"), indent=2))
