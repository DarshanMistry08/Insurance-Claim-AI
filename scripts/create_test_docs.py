"""
scripts/create_test_docs.py — Generate Specimen Test Images for Step 8 Cross-Type Regression Testing Matrix.

Fixtures created:
  1. test_star_health_policy_schedule.png (Star Health Policy Schedule with exact Patna fields)
  2. test_claim_form.png (Claim Form with genuine incident data)
  3. test_medical_bill.png (Medical Bill Invoice)
  4. test_kyc_id_proof.png (KYC ID Proof - PAN / Aadhaar)
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

out_dir = Path("data/uploads")
out_dir.mkdir(parents=True, exist_ok=True)

def create_image(filename: str, lines: list[str], title: str):
    width, height = 950, 1150
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Header bar
    draw.rectangle([0, 0, width, 80], fill=(240, 243, 246))
    draw.text((30, 25), title, fill=(15, 23, 42), font_size=24)

    y = 110
    for line in lines:
        if line.startswith("=="):
            draw.line([(30, y), (width - 30, y)], fill=(200, 200, 200), width=2)
            y += 20
        elif line.startswith("HEADER:"):
            draw.text((30, y), line.replace("HEADER:", "").strip(), fill=(30, 58, 138), font_size=18)
            y += 35
        else:
            draw.text((30, y), line, fill=(30, 41, 59), font_size=15)
            y += 28

    out_path = out_dir / filename
    img.save(out_path)
    print(f"Created specimen test document: {out_path}")

# 1. Star Health Family Optima Policy Schedule (with exact fields from specification)
star_health_lines = [
    "HEADER: STAR HEALTH AND ALLIED INSURANCE COMPANY LIMITED",
    "POLICY SCHEDULE - FAMILY OPTIMA HEALTH INSURANCE",
    "==",
    "Policy Number: P/700002/01/2023/007530",
    "Previous Policy No: P/700002/01/2022/006120",
    "Proposer Name: ANANT KUMAR",
    "Proposer Address: 3A, Kunwar Singh Chowk, Danapur, Near Saguna More, Patna, Bihar - 801503",
    "Issuing Office Address: Star Health House, No. 1 New Tank Street, Nungambakkam, Chennai 600034",
    "Sum Insured: Rs 3,00,000",
    "Gross Premium: Rs 14,500",
    "Period of Insurance From: 03/07/2022 To: 02/07/2023",
    "Scheme Description: Family Optima Floater Plan",
    "Insured Persons: ANANT KUMAR (38), SUNITA KUMARI (35)",
    "Intermediary Name: Delhi Telesales",
    "Policy Type: Health Insurance Floater",
]
create_image("test_star_health_policy_schedule.png", star_health_lines, "POLICY SCHEDULE - STAR HEALTH")

# 2. Real Claim Form
claim_form_lines = [
    "HEADER: INSURANCE CLAIM NOTIFICATION FORM",
    "==",
    "Claimant Name: John Doe",
    "Policy Number: POL-99887766",
    "Incident Date: 12/05/2024",
    "Claim Filing Date: 15/05/2024",
    "Incident Type: Water Damage",
    "Incident Description: Burst pipe in master bathroom caused flooding and hardwood floor damage.",
    "Amount Claimed: $12,500",
    "Claimant Address: 742 Evergreen Terrace, Springfield",
    "Adjuster Name: Michael Scott",
    "Supporting Doc Count: 4",
]
create_image("test_claim_form.png", claim_form_lines, "INSURANCE CLAIM FORM")

# 3. Medical Bill Invoice
medical_bill_lines = [
    "HEADER: ST. JUDE MEMORIAL HOSPITAL - TAX INVOICE",
    "==",
    "Patient Name: Anita Roy",
    "Hospital Name: St. Jude Memorial Hospital",
    "Admission Date: 10/06/2024",
    "Discharge Date: 14/06/2024",
    "Diagnosis Code: ICD-10 J18.9 (Pneumonia, unspecified)",
    "Itemized Charges: ICU Bed ($3,000), Nursing Care ($1,200), Pharmacy & IV ($1,500), Lab Tests ($800)",
    "Total Amount: $6,500",
]
create_image("test_medical_bill.png", medical_bill_lines, "MEDICAL BILL INVOICE")

# 4. KYC ID Proof
kyc_id_lines = [
    "HEADER: INCOME TAX DEPARTMENT - GOVT OF INDIA",
    "PERMANENT ACCOUNT NUMBER CARD (PAN)",
    "==",
    "ID Type: PAN Card",
    "ID Number: ABCDE1234F",
    "Full Name: ANANT KUMAR",
    "Date of Birth: 15/08/1986",
    "Address: 3A, Kunwar Singh Chowk, Danapur, Near Saguna More, Patna, Bihar - 801503",
]
create_image("test_kyc_id_proof.png", kyc_id_lines, "KYC ID PROOF")
