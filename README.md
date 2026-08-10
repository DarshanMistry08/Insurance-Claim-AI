# Insurance Claims AI Pipeline

> Multi-agent system for automated insurance claim processing, field extraction, policy verification, and fraud detection.

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Docker Desktop** (for Postgres + pgvector)
- **HuggingFace account** (for dataset access — some datasets require auth)

### 1. Clone & set up Python environment

```bash
cd demo2
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 2. Start the database

```bash
docker-compose up -d
```

This spins up Postgres 16 with pgvector, creates the schema, and exposes port 5432.

Verify it's running:
```bash
docker exec -it claims-postgres psql -U claims -d claims_db -c "\dt"
```

### 3. Pull datasets

```bash
# FUNSD — scanned forms with entity annotations
python scripts/pull_funsd.py --limit 30

# DocVQA — document QA pairs (optional, for benchmarking)
python scripts/pull_docvqa.py --limit 50
```

### 4. Inspect & label

```bash
# Browse downloaded documents
python scripts/inspect_dataset.py --dataset funsd --limit 10

# Open data/golden_set.csv in Excel/Sheets and fill in ground truth values
# Then seed into the database:
python scripts/seed_golden_set.py
```

## Project Structure

```
demo2/
├── SPEC.md                    # ← North-star spec (claim definition, fields, flags)
├── docker-compose.yml         # Postgres + pgvector
├── requirements.txt           # Python dependencies
├── db/
│   └── init.sql               # Database schema (auto-runs on first docker-compose up)
├── data/
│   ├── golden_set.csv         # Hand-labeled ground truth (you fill this in)
│   ├── funsd/                 # Downloaded FUNSD images + annotations
│   └── docvqa/                # Downloaded DocVQA images + QA pairs
└── scripts/
    ├── pull_funsd.py          # Download FUNSD dataset
    ├── pull_docvqa.py         # Download DocVQA dataset
    ├── inspect_dataset.py     # Browse & inspect downloaded data
    └── seed_golden_set.py     # Load golden labels into Postgres
```

## Spec Overview

See [SPEC.md](SPEC.md) for the full one-page spec. In short:

- **10 extraction fields**: claimant_name, policy_number, incident_date, claim_filing_date, incident_type, incident_description, amount_claimed, claimant_address, adjuster_name, supporting_doc_count
- **4 flag conditions**: Amount mismatch, expired policy, duplicate claim, missing required fields
- **Scope**: English-only, batch processing, no PII compliance (portfolio project)

## Roadmap

| Phase | Timeline | Focus |
|-------|----------|-------|
| **0** | Day 1 | Spec document ✅ |
| **1** | Week 1 | Data + storage (this setup) |
| **2** | Weeks 2–3 | Vertical slice (one doc type, end-to-end) |
| **3** | Weeks 4–6 | LangGraph multi-agent orchestration |
| **4** | Weeks 7–8 | Eval harness (pytest + Langfuse) |
| **5** | Weeks 9–12 | FastAPI, Docker, demo, blog post |

## Dataset & Model Retraining Recommendations
To improve extraction accuracy and fraud detection, we recommend the following dataset and model improvements:

1. **Dataset Auditing & Fixes (Data-Centric AI)**
   - Run `python scripts/dataset_audit.py` to identify annotation issues (e.g., missing labels, bounding boxes < 10px).
   - Fix class imbalances. FUNSD is heavily skewed towards the 'O' (Other) class. Ensure critical fields like `policy_number` are heavily represented in your training subset.
   - Use Label Studio to correct systematic mislabeling of mandatory fields.

2. **Preprocessing Pipeline**
   - **Deskewing**: Donut struggles with rotated text. Implement auto-deskewing.
   - **Contrast Enhancement**: Use OpenCV (`cv2.adaptiveThreshold`) to clean faded ink scans before sending them to the model.

3. **Fine-Tuning Strategy**
   - **Domain Adaptation**: The Donut model is currently fine-tuned on general DocVQA. Fine-tune it exclusively on Insurance Claim forms to learn spatial structures (where names and amounts typically appear).
   - **Augmentation**: Train on augmented images with noise, blur, and slight rotations to mimic mobile phone uploads.
