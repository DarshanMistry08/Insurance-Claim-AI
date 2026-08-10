# Insurance Claims Processing — Project Spec (v1.0)

> **One-page north-star document. All scope decisions trace back here.**

---

## 1. What Is a "Claim Document"?

A claim document is any artifact submitted as part of a first-notice-of-loss (FNOL) insurance claim. For this project we focus on three categories:

| Type | Description | Example |
|------|-------------|---------|
| **Claim Form** | Structured/semi-structured scanned form (ACORD-style or carrier-specific) containing policyholder info, incident details, and amounts | ACORD 1 — Property Loss Notice |
| **Supporting Table/Invoice** | Itemised repair estimates, medical bills, or receipts attached to the claim | Auto body shop repair estimate |
| **Photo/Image** | Photographic evidence of damage (vehicle, property, injury) | Hail damage photo |

**Primary focus for Phase 1–3:** Claim Forms (structured extraction).  
Photos and invoices become relevant in Phase 4+ for multi-modal validation.

---

## 2. Extraction Fields (10)

These are the fields the extraction agent must pull from every claim form:

| # | Field Name | Type | Example Value |
|---|-----------|------|---------------|
| 1 | `claimant_name` | string | "Jane A. Martinez" |
| 2 | `policy_number` | string | "HO-2024-0847291" |
| 3 | `incident_date` | date (ISO 8601) | "2025-11-03" |
| 4 | `claim_filing_date` | date (ISO 8601) | "2025-11-10" |
| 5 | `incident_type` | enum | fire · theft · water · collision · liability · other |
| 6 | `incident_description` | string (≤500 chars) | "Kitchen fire caused by unattended stove…" |
| 7 | `amount_claimed` | decimal (USD) | 34250.00 |
| 8 | `claimant_address` | string | "412 Oak Lane, Springfield, IL 62704" |
| 9 | `adjuster_name` | string \| null | "Robert Chen" |
| 10 | `supporting_doc_count` | integer | 3 |

**Evaluation metric:** Per-field exact-match accuracy and overall F1 across the golden set.

---

## 3. Flag Conditions (4)

The reasoning agent checks every processed claim against these rules:

| # | Flag | Logic | Severity |
|---|------|-------|----------|
| 1 | **Amount Mismatch** | `amount_claimed` > coverage limit in the matched policy document | 🔴 High |
| 2 | **Expired Policy** | `incident_date` falls outside the policy's effective → expiration date range | 🔴 High |
| 3 | **Duplicate Claim** | Cosine similarity > 0.92 against any previously processed claim (same claimant + similar incident description) | 🟡 Medium |
| 4 | **Missing Required Fields** | Any of `claimant_name`, `policy_number`, `incident_date`, `amount_claimed` is null or unparseable | 🟡 Medium |

**Evaluation metric:** Flag precision and recall against hand-labeled flag annotations in the golden set.

---

## Scope Boundaries (What This Project Is **Not**)

- Not a production deployment — no PII handling, no HIPAA/SOC2 compliance
- Not multi-language — English documents only
- Not real-time — batch processing is fine
- Not a full claims management system — no workflow UI, no adjuster assignment
