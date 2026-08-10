"""
scripts/generate_synthetic_claims.py
Generate 30 synthetic insurance claim form PNG images for the golden evaluation set.

Each image is a realistic-looking claim form with the exact field values from
data/golden_set.json, making them ground-truth-verifiable by the extraction agent.

Usage:
    python scripts/generate_synthetic_claims.py
    python scripts/generate_synthetic_claims.py --out-dir data/synthetic_claims/images

Output: 30 PNG files named claim_001.png ... claim_030.png
"""

import os
import json
import sys
from pathlib import Path
import textwrap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_set.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "synthetic_claims" / "images"

# ── PIL is the only dependency needed here ───────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: Pillow not installed. Run: pip install Pillow")
    sys.exit(1)


def _try_load_font(size: int):
    """Try to load a monospace font; fall back to PIL default."""
    candidates = [
        "cour.ttf",          # Windows Courier New
        "CourierNew.ttf",
        "DejaVuSansMono.ttf",
        "LiberationMono-Regular.ttf",
        "Courier New Bold.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_claim_image(claim: dict, out_path: Path) -> None:
    """Render a single claim record as a PNG form image."""
    gt = claim["ground_truth"]
    doc_id = claim["doc_id"]

    W, H = 900, 1100
    BG = (252, 252, 250)         # off-white paper
    HEADER_BG = (30, 60, 114)    # dark navy
    LINE_COLOR = (180, 180, 185)
    TEXT_DARK = (20, 20, 30)
    TEXT_GRAY = (100, 100, 110)
    ACCENT = (30, 80, 160)

    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    font_title  = _try_load_font(24)
    font_sub    = _try_load_font(14)
    font_label  = _try_load_font(11)
    font_value  = _try_load_font(15)
    font_footer = _try_load_font(10)

    # ── Header bar ──────────────────────────────────────────────
    draw.rectangle([(0, 0), (W, 90)], fill=HEADER_BG)
    draw.text((40, 18), "MERIDIAN INSURANCE CORPORATION", font=font_title, fill=(255, 255, 255))
    draw.text((40, 52), "INSURANCE CLAIM FORM  •  Form IC-2024-A", font=font_sub, fill=(180, 200, 240))
    draw.text((680, 34), f"Doc: {doc_id}", font=font_label, fill=(160, 190, 230))

    # ── Form boundary ───────────────────────────────────────────
    draw.rectangle([(30, 100), (W - 30, H - 40)], outline=LINE_COLOR, width=2)

    # ── Section helper ───────────────────────────────────────────
    def section_header(y, title):
        draw.rectangle([(30, y), (W - 30, y + 26)], fill=(230, 235, 245))
        draw.text((42, y + 5), title.upper(), font=font_label, fill=ACCENT)
        return y + 34

    def field_row(y, label, value, indent=50, label_w=220):
        draw.text((indent, y), label + ":", font=font_label, fill=TEXT_GRAY)
        # underline
        draw.line([(indent + label_w, y + 17), (W - 50, y + 17)], fill=LINE_COLOR, width=1)
        # wrap long values
        wrapped = textwrap.shorten(str(value), width=60, placeholder="...")
        draw.text((indent + label_w, y), wrapped, font=font_value, fill=TEXT_DARK)
        return y + 36

    y = 110
    y = section_header(y, "Policyholder Information")
    y = field_row(y, "Claimant Name",      gt.get("claimant_name", ""))
    y = field_row(y, "Policy Number",      gt.get("policy_number", ""))
    y = field_row(y, "Claimant Address",   gt.get("claimant_address", ""))
    y += 10

    y = section_header(y, "Incident Details")
    y = field_row(y, "Incident Type",      gt.get("incident_type", "").upper())
    y = field_row(y, "Incident Date",      gt.get("incident_date", ""))
    y = field_row(y, "Claim Filing Date",  gt.get("claim_filing_date", ""))

    # Description — multi-line
    desc = gt.get("incident_description", "")
    draw.text((50, y), "Incident Description:", font=font_label, fill=TEXT_GRAY)
    y += 18
    lines = textwrap.wrap(desc, width=80)
    for line in lines[:4]:
        draw.text((70, y), line, font=font_value, fill=TEXT_DARK)
        y += 22
    y += 10

    y = section_header(y, "Financial Information")
    y = field_row(y, "Amount Claimed",     gt.get("amount_claimed", ""))
    y += 10

    y = section_header(y, "Adjuster Assignment")
    y = field_row(y, "Adjuster Name",      gt.get("adjuster_name", ""))
    y = field_row(y, "Supporting Documents", gt.get("supporting_doc_count", "") + " attached")
    y += 20

    # ── Signature line ───────────────────────────────────────────
    draw.line([(50, y + 30), (350, y + 30)], fill=TEXT_DARK, width=1)
    draw.line([(450, y + 30), (W - 50, y + 30)], fill=TEXT_DARK, width=1)
    draw.text((50,  y + 34), "Insured Signature", font=font_footer, fill=TEXT_GRAY)
    draw.text((450, y + 34), "Date", font=font_footer, fill=TEXT_GRAY)

    # ── Footer ───────────────────────────────────────────────────
    draw.rectangle([(0, H - 36), (W, H)], fill=(245, 245, 248))
    draw.text((40, H - 24), "CONFIDENTIAL — INSURANCE CLAIM DOCUMENT — DO NOT DISTRIBUTE", font=font_footer, fill=(140, 140, 150))
    draw.text((W - 160, H - 24), f"Page 1 of 1", font=font_footer, fill=(140, 140, 150))

    img.save(str(out_path), "PNG", dpi=(150, 150))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic claim form images.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for PNG images")
    parser.add_argument("--golden", default=str(GOLDEN_PATH), help="Path to golden_set.json")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.golden, encoding="utf-8") as f:
        data = json.load(f)

    claims = data["claims"]
    print(f"Generating {len(claims)} claim form images → {out_dir}")

    for i, claim in enumerate(claims, 1):
        doc_id = claim["doc_id"]
        out_path = out_dir / f"{doc_id}.png"
        try:
            draw_claim_image(claim, out_path)
            print(f"  [{i:02d}/{len(claims)}] {doc_id}.png")
        except Exception as e:
            print(f"  [{i:02d}/{len(claims)}] ERROR for {doc_id}: {e}")

    print(f"\nDone. {len(claims)} images in {out_dir}")


if __name__ == "__main__":
    main()
