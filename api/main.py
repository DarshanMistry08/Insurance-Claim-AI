"""
api/main.py — FastAPI backend for the Claims AI Pipeline.

Endpoints:
  POST /claims/process  — Upload a claim image, run the LangGraph pipeline, return JSON result.
  GET  /claims/{doc_id} — Retrieve a previously processed claim from Postgres.
  GET  /health          — Health check for Ollama + Postgres.
"""

import os
import shutil
import uuid
import json
from pathlib import Path

import psycopg2
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ClaimResult, HealthResponse

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "claims_db"),
    "user": os.getenv("DB_USER", "claims"),
    "password": os.getenv("DB_PASSWORD", ""),
}

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

app = FastAPI(
    title="Claims AI Pipeline API",
    description="Multi-agent insurance claim processing powered by LangGraph + Ollama",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check connectivity to Ollama and Postgres."""
    ollama_status = "ok"
    postgres_status = "ok"

    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if r.status_code != 200:
            ollama_status = "unavailable"
    except Exception:
        ollama_status = "unavailable"

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.close()
    except Exception:
        postgres_status = "unavailable"

    return HealthResponse(
        status="ok" if ollama_status == "ok" and postgres_status == "ok" else "degraded",
        ollama=ollama_status,
        postgres=postgres_status,
    )


@app.post("/claims/process", response_model=ClaimResult)
async def process_claim(file: UploadFile = File(...)):
    """
    Upload a claim document image (PNG/JPG).
    Runs the full LangGraph pipeline and returns structured results.
    """
    # Save uploaded image
    ext = Path(file.filename).suffix or ".png"
    doc_id = f"upload_{uuid.uuid4().hex[:8]}"
    image_path = UPLOAD_DIR / f"{doc_id}{ext}"

    with image_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Import graph lazily to avoid long startup time
    from pipeline.graph import app as graph_app

    initial_state = {
        "doc_id": doc_id,
        "image_path": str(image_path),
        "extracted_fields": {},
        "policy_context": "",
        "is_duplicate": False,
        "fraud_score": "",
        "final_summary": "",
        "flags": [],
    }

    try:
        result = graph_app.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    return ClaimResult(
        doc_id=doc_id,
        document_type=result.get("document_type", "unknown"),
        document_type_confidence=result.get("document_type_confidence", 0.0),
        classification_status=result.get("classification_status", "certain"),
        extracted_fields=result.get("extracted_fields", {}),
        policy_context=result.get("policy_context", ""),
        is_duplicate=result.get("is_duplicate", False),
        fraud_score=result.get("fraud_score", ""),
        flags=result.get("flags", []),
        final_summary=result.get("final_summary", {}),
        agent_trace=result.get("agent_trace", []),
        verification_verdict=result.get("verification_verdict", {}),
        fraud_result=result.get("fraud_result", {}),
        recommendation=result.get("final_summary", {}).get("recommendation", "") if isinstance(result.get("final_summary"), dict) else "",
        confidence_score=result.get("final_summary", {}).get("confidence_score", 0.0) if isinstance(result.get("final_summary"), dict) else 0.0,
        citations=result.get("final_summary", {}).get("citations", []) if isinstance(result.get("final_summary"), dict) else [],
        retrieved_clauses=result.get("retrieved_clauses", []),
    )


@app.get("/claims/{doc_id}", response_model=ClaimResult)
def get_claim(doc_id: str):
    """Retrieve a previously processed claim from Postgres."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT pc.extracted_fields, pc.flags, pc.llm_reasoning, d.file_path
               FROM processed_claims pc
               JOIN documents d ON pc.document_id = d.document_id
               WHERE d.file_path LIKE %s
               LIMIT 1;""",
            (f"%{doc_id}%",),
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    if not row:
        raise HTTPException(status_code=404, detail=f"Claim '{doc_id}' not found.")

    extracted_fields, flags, llm_reasoning, file_path = row

    # flags is a JSONB array of flag strings stored directly
    flags_list = flags if isinstance(flags, list) else []

    return ClaimResult(
        doc_id=doc_id,
        extracted_fields=extracted_fields or {},
        policy_context="",
        is_duplicate=False,
        fraud_score="",
        flags=flags_list,
        final_summary=llm_reasoning or "",
    )
