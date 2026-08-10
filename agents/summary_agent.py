"""
agents/summary_agent.py — Adjuster Summary Agent (Node 5 of 5).

Generates a structured AdjusterSummary:
  - recommendation: APPROVE | FLAG_FOR_REVIEW | REJECT
  - confidence_score: 0.0–1.0
  - summary_text: Grounded Markdown narrative
  - citations: [{field_ref, clause_ref, rationale}]
"""

from __future__ import annotations
import os
import re
import json
import time
import logging
import requests

import psycopg2

from agents.state import (
    ClaimState, AdjusterSummary, Citation, NodeLog
)
from agents.tracing import NodeTracer

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME", "claims_db"),
    "user":     os.getenv("DB_USER", "claims"),
    "password": os.getenv("DB_PASSWORD", ""),
}


def _derive_recommendation(flags: list[dict], verification_verdict: dict, fraud_result: dict, doc_type: str) -> tuple[str, float]:
    """Rule-based recommendation derived from document type, flags, and verdicts."""
    if doc_type != "claim_form":
        # Non-claim documents (e.g. policy_schedule) do not get flagged for claim fraud
        high_flags = [f for f in flags if f.get("severity") == "HIGH"]
        if high_flags:
            return "FLAG_FOR_REVIEW", 0.70
        return "APPROVE", 0.90

    flag_codes = {f.get("code", "") for f in flags}
    coverage_status = verification_verdict.get("coverage_status", "uncertain") if verification_verdict else "uncertain"
    fraud_label = fraud_result.get("label", "legitimate") if fraud_result else "legitimate"

    high_severity_flags = [f for f in flags if f.get("severity") == "HIGH"]

    if fraud_label == "fraudulent" or "DUPLICATE_CLAIM" in flag_codes:
        return "REJECT", 0.92
    if "COVERAGE_EXCLUDED" in flag_codes or "OUTSIDE_POLICY_PERIOD" in flag_codes:
        return "REJECT", 0.88
    if len(high_severity_flags) >= 2:
        return "FLAG_FOR_REVIEW", 0.80
    if any("EXCEEDS" in c for c in flag_codes) or fraud_label == "suspicious":
        return "FLAG_FOR_REVIEW", 0.72
    if coverage_status == "covered":
        return "APPROVE", 0.85
    if coverage_status == "uncertain":
        return "FLAG_FOR_REVIEW", 0.60
    return "APPROVE", 0.75


def _build_citations(flags: list[dict], findings: list[dict]) -> list[Citation]:
    citations: list[Citation] = []
    for flag in flags:
        citations.append(Citation(
            field_ref=flag.get("field_ref", ""),
            clause_ref=flag.get("clause_ref", ""),
            rationale=f"Flag [{flag.get('code', '')}]: {flag.get('description', '')}",
        ))
    for finding in findings:
        if finding.get("result") not in ("covered", "within_limit"):
            citations.append(Citation(
                field_ref=finding.get("field_ref", ""),
                clause_ref=finding.get("clause_ref", ""),
                rationale=f"Verification finding: {finding.get('evidence', '')}",
            ))
    return citations


def _generate_summary_text(
    doc_type: str,
    extracted_fields: dict,
    flags: list[dict],
    verification_verdict: dict,
    fraud_result: dict,
    recommendation: str,
    retrieved_clauses: list[dict],
) -> str:
    """Ask Ollama to write a Grounded, Type-Aware Summary."""
    claim_json = json.dumps(extracted_fields, indent=2)

    if doc_type != "claim_form":
        field_lines = []
        for k, v in extracted_fields.items():
            val = v.get("value") if isinstance(v, dict) else str(v)
            if val:
                label = k.replace("_", " ").title()
                field_lines.append(f"- {label}: {val}")

        extracted_formatted = "\n".join(field_lines) if field_lines else "None"
        return (
            f"This document is classified as a {doc_type} and does not contain claim or incident details.\n\n"
            f"Extracted fields:\n\n{extracted_formatted}"
        )
    else:
        retrieved_chunks = "\n".join(
            f"[{c.get('clause_id', '')}]: {c.get('chunk_text', '')}"
            for c in retrieved_clauses
        ) or "No clauses retrieved."
        assessment_dict = {
            "fraud_result": fraud_result,
            "flags": flags,
            "recommendation": recommendation,
            "verification_verdict": verification_verdict
        }
        fraud_json = json.dumps(assessment_dict, indent=2)

        prompt = f"""SYSTEM:
You are a claims analyst assistant. You will be given (1) extracted claim fields as JSON, 
(2) retrieved policy context, and (3) fraud flags. You must write a summary using ONLY 
the specific values provided below. Do not use generic placeholder language. If a field 
is "NOT_FOUND" or "N/A", explicitly say that field is missing — do not invent or omit silently.

Respond in this exact structure:
1. Claim Overview (reference the actual claimant name, incident type, and amount — verbatim from input)
2. Missing Fields (list any NOT_FOUND / N/A fields by name)
3. Policy Coverage Match (state whether the retrieved context actually covers this incident_type — quote which coverage clause, or say "no matching clause found")
4. Fraud Risk (restate the confidence score and reason passed in)
5. Recommendation (FLAG FOR REVIEW / APPROVE / DENY)

USER:
Claim fields: {claim_json}
Retrieved policy context: {retrieved_chunks}
Fraud assessment: {fraud_json}"""

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}},
            timeout=90,
        )
        resp.raise_for_status()
        summary_out = resp.json().get("response", "").strip()

        # STEP 6 Post-generation Check: Rejects summaries that hallucinate claim details for non-claim docs
        if doc_type != "claim_form":
            lowered = summary_out.lower()
            if any(term in lowered for term in ["water damage", "fire damage", "incident date", "amount claimed: $"]):
                summary_out = f"1. Document Overview\nThis document is a classified {doc_type} and does not contain claim/incident details.\n\n2. Extracted Data\n{claim_json}\n\n3. Recommendation\n{recommendation}"

        return summary_out
    except Exception as e:
        logger.error("summary: LLM call failed: %s", e)
        return f"Document Type: {doc_type}\nRecommendation: {recommendation}"


def _persist_to_db(state: ClaimState, summary: AdjusterSummary, latency_ms: float) -> None:
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO processed_claims
              (document_id, extracted_fields, flags, llm_reasoning, processing_time_ms)
            SELECT d.document_id, %s::jsonb, %s::jsonb, %s, %s
            FROM documents d WHERE d.file_path LIKE %s
            ON CONFLICT DO NOTHING;
            """,
            (
                json.dumps(state.get("extracted_fields", {})),
                json.dumps(state.get("flags", [])),
                summary.summary_text,
                int(latency_ms),
                f"%{state['doc_id']}%",
            ),
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.warning("summary: DB persist failed: %s", e)


def node_summary(state: ClaimState) -> dict:
    doc_id = state["doc_id"]
    doc_type = state.get("document_type", "unknown")
    extracted_fields = state.get("extracted_fields", {})
    flags = state.get("flags", [])
    verification_verdict = state.get("verification_verdict", {})
    fraud_result = state.get("fraud_result", {})
    retrieved_clauses = state.get("retrieved_clauses", [])

    with NodeTracer("summary_agent", trace_id=doc_id, inputs={"doc_id": doc_id, "num_flags": len(flags)}) as tracer:
        t0 = time.perf_counter()
        error_msg = ""

        recommendation, confidence = _derive_recommendation(flags, verification_verdict, fraud_result, doc_type)
        findings = (verification_verdict.get("findings", []) if isinstance(verification_verdict, dict) else [])
        citations = _build_citations(flags, findings)

        summary_text = _generate_summary_text(
            doc_type, extracted_fields, flags, verification_verdict, fraud_result, recommendation, retrieved_clauses
        )

        summary = AdjusterSummary(
            recommendation=recommendation,
            confidence_score=round(confidence, 4),
            summary_text=summary_text,
            citations=citations,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        _persist_to_db(state, summary, latency_ms)

        log = NodeLog(
            node_name="summary_agent",
            latency_ms=round(latency_ms, 1),
            langfuse_span_id=tracer.span_id,
            error=error_msg,
        )
        tracer.end(outputs={"recommendation": recommendation, "confidence": confidence})

    return {
        "final_summary": summary.model_dump(),
        "final_summary_text": summary_text,
        "agent_trace": [log.model_dump()],
    }
