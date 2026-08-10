"""
scripts/test_extract_direct.py — Direct node_extract test on upload_eeffefc9.webp
"""

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path
from agents.extract_agent import node_extract

img_path = Path("data/uploads/upload_eeffefc9.webp")

state = {
    "doc_id": "test_direct_001",
    "image_path": str(img_path),
    "extracted_fields": {},
    "flags": []
}

print(f"Executing node_extract directly on {img_path}...")
res = node_extract(state)

print("\n" + "="*80)
print(f"DOCUMENT TYPE: {res.get('document_type')} (Confidence: {res.get('document_type_confidence')})")
print("="*80)
print("EXTRACTED FIELDS:")
print(json.dumps(res.get("extracted_fields"), indent=2))
