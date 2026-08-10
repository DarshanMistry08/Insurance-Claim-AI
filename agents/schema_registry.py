"""
agents/schema_registry.py — Schema Registry & Dedicated Prompt Modules per Document Type.

Implements STEP 2:
  - Isolated modules/dicts for field schemas AND extraction prompts per document_type.
  - Automated test enforcement: Every field in every schema MUST have a non-empty extraction prompt.
"""

from typing import Dict, List, Any, Optional

DOCUMENT_TYPES = [
    "policy_schedule",
    "claim_form",
    "medical_bill_invoice",
    "discharge_summary",
    "kyc_id_proof",
    "claims_history_record",
    "unknown"
]

# Classification markers (keywords & patterns)
CLASSIFICATION_MARKERS: Dict[str, List[str]] = {
    "policy_schedule": [
        "policy schedule", "schedule of insurance", "certificate of insurance",
        "period of insurance", "sum insured", "gross premium", "proposer",
        "insured persons", "intermediary", "policy type", "star health", "optima"
    ],
    "claim_form": [
        "claim form", "claim notification", "notice of claim", "claimant name",
        "date of loss", "amount claimed", "incident description", "cause of loss",
        "claim filing", "adjuster"
    ],
    "medical_bill_invoice": [
        "tax invoice", "final bill", "medical bill", "hospital bill",
        "admission date", "discharge date", "itemized charges", "net payable",
        "diagnosis", "icd code", "pharmacy bill", "consultation fee"
    ],
    "discharge_summary": [
        "discharge summary", "discharge card", "course in hospital",
        "treatment given", "clinical summary", "condition on discharge", "final diagnosis"
    ],
    "kyc_id_proof": [
        "income tax department", "permanent account number", "pan card",
        "unique identification authority", "aadhaar", "passport", "driving licence",
        "date of birth", "dob", "father's name"
    ],
    "claims_history_record": [
        "claim history", "claims statement", "prior claims", "track record", "loss history"
    ]
}

# ─── ISOLATED FIELD SCHEMAS ───────────────────────────────────── #

POLICY_SCHEDULE_SCHEMA = {
    "policy_number": "Policy Number or Policy No. or Schedule Number",
    "previous_policy_number": "Previous Policy No. or Prior Policy Number",
    "proposer_name": "Name of Proposer or Policyholder or Primary Insured",
    "proposer_address": "Address of the Proposer or Policyholder",
    "issuing_office_address": "Address of the Issuing Office or Branch Office",
    "sum_insured": "Sum Insured or Total Coverage Amount",
    "premium": "Gross Premium or Total Premium Amount",
    "period_of_insurance_from": "Policy Start/Inception Date (From)",
    "period_of_insurance_to": "Policy Expiry/End Date (To)",
    "scheme_description": "Scheme or Plan or Product Description",
    "insured_persons": "Covered Insured Persons or Family Members List",
    "intermediary_name": "Agent or Broker or Intermediary Name",
    "policy_type": "Policy Type (e.g. Health, Motor, Property)"
}

CLAIM_FORM_SCHEMA = {
    "claimant_name": "Claimant Name or Insured Name submitting claim",
    "policy_number": "Policy Number for this claim",
    "incident_date": "Exact Incident Date or Date of Loss",
    "claim_filing_date": "Claim Filing Date or Report Date",
    "incident_type": "Incident Type, Peril, or Cause of Loss",
    "incident_description": "Incident Description or Loss Narrative",
    "amount_claimed": "Amount Claimed or Total Loss Claimed",
    "claimant_address": "Claimant Address",
    "adjuster_name": "Adjuster Name or Assigned Claims Handler",
    "supporting_doc_count": "Number of attached supporting documents"
}

MEDICAL_BILL_SCHEMA = {
    "patient_name": "Patient Full Name",
    "hospital_name": "Hospital or Clinic Name",
    "admission_date": "Date of Admission",
    "discharge_date": "Date of Discharge",
    "itemized_charges": "Itemized Charges or Billed Medical Services",
    "total_amount": "Total Amount or Net Payable",
    "diagnosis_code": "Diagnosis Description or ICD Code"
}

DISCHARGE_SUMMARY_SCHEMA = {
    "patient_name": "Patient Full Name",
    "hospital_name": "Hospital or Medical Facility Name",
    "admission_date": "Date of Admission",
    "discharge_date": "Date of Discharge",
    "diagnosis": "Final Diagnosis or Clinical Condition on Discharge",
    "treatment_given": "Treatment Given or Surgical Procedure Performed"
}

KYC_ID_PROOF_SCHEMA = {
    "id_type": "ID Type (e.g., PAN Card, Aadhaar, Passport, Driving License)",
    "id_number": "Card or Document ID Number",
    "full_name": "Full Name printed on ID",
    "dob": "Date of Birth",
    "address": "Address printed on ID"
}

CLAIMS_HISTORY_SCHEMA = {
    "policy_number": "Policy Number",
    "insured_name": "Insured Person Name",
    "prior_claims": "Prior Claim Details or Past Claims Amount"
}

SCHEMA_REGISTRY: Dict[str, Dict[str, str]] = {
    "policy_schedule": POLICY_SCHEDULE_SCHEMA,
    "claim_form": CLAIM_FORM_SCHEMA,
    "medical_bill_invoice": MEDICAL_BILL_SCHEMA,
    "discharge_summary": DISCHARGE_SUMMARY_SCHEMA,
    "kyc_id_proof": KYC_ID_PROOF_SCHEMA,
    "claims_history_record": CLAIMS_HISTORY_SCHEMA,
    "unknown": {"document_summary": "Summary of unclassified document"}
}


# ─── DEDICATED EXTRACTION PROMPT REGISTRY ──────────────────────── #

POLICY_SCHEDULE_EXTRACTION_PROMPTS: Dict[str, str] = {
    "policy_number": "Find the literal printed label 'Policy No.' or 'Policy Number' or 'Schedule No.' and extract the alphanumeric value immediately following it. Do not confuse with previous policy number.",
    "previous_policy_number": "Find the label 'Previous Policy No.' or 'Prior Policy No.' and extract the exact value following it.",
    "proposer_name": "Find the label 'Name of Proposer' or 'Proposer Name' or 'Insured Name' and extract the full person name.",
    "proposer_address": "Find the label 'Address of Proposer' or 'Proposer Address' or 'Insured Address' and extract the complete postal address. Do NOT mix with Issuing Office address.",
    "issuing_office_address": "Find the label 'Issuing Office Address' or 'Branch Address' and extract the company issuing office address.",
    "sum_insured": "Find the label 'Sum Insured' or 'Total Sum Insured' or 'Limit of Indemnity' and extract the exact currency value (e.g., Rs. 3,00,000).",
    "premium": "Find the label 'Gross Premium' or 'Total Premium' or 'Net Premium' and extract the exact premium amount.",
    "period_of_insurance_from": "Find the label 'Period of Insurance' or 'Policy Period' and extract the start/inception date ('From' or first date).",
    "period_of_insurance_to": "Find the label 'Period of Insurance' or 'Policy Period' and extract the expiry/end date ('To' or second date).",
    "scheme_description": "Find the label 'Scheme Description' or 'Plan Name' or 'Product Name' and extract the policy plan name.",
    "insured_persons": "Find the list/table of 'Insured Persons' or 'Persons Covered' and extract all covered member names and ages.",
    "intermediary_name": "Find the label 'Intermediary Name' or 'Agent Name' or 'Broker Code/Name' and extract the broker/agent name.",
    "policy_type": "Find the policy header or type label (e.g., Health, Motor, Commercial) and extract the policy category."
}

CLAIM_FORM_EXTRACTION_PROMPTS: Dict[str, str] = {
    "claimant_name": "Find the label 'Claimant Name' or 'Insured Name' submitting the claim and extract the full name.",
    "policy_number": "Find the label 'Policy No.' or 'Policy Number' on the claim form and extract the value.",
    "incident_date": "Find the label 'Incident Date' or 'Date of Loss' and extract the exact date the damage occurred.",
    "claim_filing_date": "Find the label 'Claim Filing Date' or 'Report Date' and extract the filing date.",
    "incident_type": "Find the label 'Incident Type' or 'Cause of Loss' or 'Peril' and extract the type of incident.",
    "incident_description": "Find the label 'Incident Description' or 'Loss Description' and extract the narrative narrative.",
    "amount_claimed": "Find the label 'Amount Claimed' or 'Total Claim Amount' and extract the monetary figure.",
    "claimant_address": "Find the label 'Claimant Address' and extract the residential address.",
    "adjuster_name": "Find the label 'Adjuster Name' or 'Claims Handler' and extract the name.",
    "supporting_doc_count": "Find the label 'Supporting Documents' or count of attachments and extract the integer count."
}

MEDICAL_BILL_EXTRACTION_PROMPTS: Dict[str, str] = {
    "patient_name": "Find the label 'Patient Name' or 'Billed To' and extract the patient full name.",
    "hospital_name": "Find the header logo or 'Hospital Name' and extract the medical facility name.",
    "admission_date": "Find the label 'Admission Date' or 'DOA' and extract the admission date.",
    "discharge_date": "Find the label 'Discharge Date' or 'DOD' and extract the discharge date.",
    "itemized_charges": "Find the table or list of billed items/services and extract itemized breakdown.",
    "total_amount": "Find the label 'Total Amount' or 'Net Payable' or 'Grand Total' and extract the final amount.",
    "diagnosis_code": "Find the label 'Diagnosis' or 'ICD Code' and extract the clinical diagnosis."
}

DISCHARGE_SUMMARY_EXTRACTION_PROMPTS: Dict[str, str] = {
    "patient_name": "Find the label 'Patient Name' and extract the name.",
    "hospital_name": "Find the facility header or 'Hospital Name' and extract the facility name.",
    "admission_date": "Find the label 'Date of Admission' and extract the date.",
    "discharge_date": "Find the label 'Date of Discharge' and extract the date.",
    "diagnosis": "Find the label 'Final Diagnosis' and extract the diagnosis text.",
    "treatment_given": "Find the label 'Treatment Given' or 'Course in Hospital' and extract the narrative."
}

KYC_ID_PROOF_EXTRACTION_PROMPTS: Dict[str, str] = {
    "id_type": "Determine the ID type from header (e.g. PAN Card, Aadhaar, Passport, Driving License).",
    "id_number": "Find the primary ID number string (e.g., 10-character PAN string, 12-digit Aadhaar).",
    "full_name": "Find the full name printed on the card.",
    "dob": "Find the label 'Date of Birth' or 'DOB' and extract the date.",
    "address": "Find the printed address block and extract the full address."
}

CLAIMS_HISTORY_EXTRACTION_PROMPTS: Dict[str, str] = {
    "policy_number": "Find the policy number on the history statement.",
    "insured_name": "Find the insured name on the history statement.",
    "prior_claims": "Find the list of prior claims or total historical payout amount."
}

EXTRACTION_PROMPT_REGISTRY: Dict[str, Dict[str, str]] = {
    "policy_schedule": POLICY_SCHEDULE_EXTRACTION_PROMPTS,
    "claim_form": CLAIM_FORM_EXTRACTION_PROMPTS,
    "medical_bill_invoice": MEDICAL_BILL_EXTRACTION_PROMPTS,
    "discharge_summary": DISCHARGE_SUMMARY_EXTRACTION_PROMPTS,
    "kyc_id_proof": KYC_ID_PROOF_EXTRACTION_PROMPTS,
    "claims_history_record": CLAIMS_HISTORY_EXTRACTION_PROMPTS,
    "unknown": {"document_summary": "Extract a general overview of the document."}
}


def get_schema_for_doc_type(doc_type: str) -> Dict[str, str]:
    return SCHEMA_REGISTRY.get(doc_type, SCHEMA_REGISTRY["unknown"])


def get_prompt_for_field(doc_type: str, field_name: str) -> str:
    prompts = EXTRACTION_PROMPT_REGISTRY.get(doc_type, {})
    return prompts.get(field_name, f"Extract the field '{field_name}' for document type '{doc_type}'.")


def classify_ocr_text(ocr_text: str) -> tuple[str, float]:
    if not ocr_text:
        return "unknown", 0.0

    text_lower = ocr_text.lower()
    scores: Dict[str, int] = {}

    for doc_type, markers in CLASSIFICATION_MARKERS.items():
        matches = sum(1 for m in markers if m in text_lower)
        if matches > 0:
            scores[doc_type] = matches

    if not scores:
        return "unknown", 0.30

    best_type = max(scores, key=scores.get)
    max_matches = scores[best_type]
    
    confidence = min(0.95, 0.50 + (max_matches * 0.10))
    
    if "schedule" in text_lower and ("sum insured" in text_lower or "policy schedule" in text_lower or "star health" in text_lower):
        if best_type != "policy_schedule":
            best_type = "policy_schedule"
            confidence = 0.95

    return best_type, round(confidence, 2)
