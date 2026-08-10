from __future__ import annotations
import os
import re
import json
import time
import logging
import requests

import psycopg2
from sentence_transformers import SentenceTransformer

from agents.state import ClaimState, FraudResult, NodeLog, Flag
from agents.tracing import NodeTracer

logger = logging.getLogger(__name__)

HF_TOKEN         = os.getenv("HF_TOKEN", "")
HF_BART_URL      = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL", "llama3.2")
DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", "0.92"))

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME", "claims_db"),
    "user":     os.getenv("DB_USER", "claims"),
    "password": os.getenv("DB_PASSWORD", ""),
}

_emb_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _emb_model
    if _emb_model is None:
        _emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _emb_model


def _field_val(extracted_fields: dict, name: str) -> str:
    f = extracted_fields.get(name, {})
    return (f.get("value", "N/A") if isinstance(f, dict) else str(f))


def _classify_via_bart(claim_text: str) -> tuple[str, float]:
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {
        "inputs": claim_text,
        "parameters": {
            "candidate_labels": [
                "legitimate insurance claim",
                "suspicious insurance claim",
                "fraudulent insurance claim",
            ],
            "multi_label": False,
        },
    }
    resp = requests.post(HF_BART_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    labels = data.get("labels", [])
    scores = data.get("scores", [])
    if not labels:
        return "legitimate", 0.5

    top_idx = 0  # sorted by score desc
    label = labels[top_idx]
    score = float(scores[top_idx]) if scores else 0.5

    if "fraudulent" in label:
        return "fraudulent", score
    if "suspicious" in label:
        return "suspicious", score
    return "legitimate", score


def _classify_via_ollama(extracted_fields: dict, flags: list, verification: dict, is_duplicate: bool) -> tuple[str, float, str]:
    simple_fields = {k: v.get("value", "N/A") for k, v in extracted_fields.items() if isinstance(v, dict)}

    prompt = f"""Assess this insurance claim for fraud risk.
Consider the extracted fields, existing flags, policy verification, and duplicate status.
Return ONLY a JSON: {{"label": "legitimate"|"suspicious"|"fraudulent", "confidence": 0.0-1.0, "reason": "Detailed 1-2 sentence explanation of why this risk level was assigned based on the data."}}

Fields: {json.dumps(simple_fields)}
Flags: {[f.get("code") for f in flags]}
Verification Status: {verification.get("coverage_status", "unknown")}
Is Duplicate: {is_duplicate}
"""

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json().get("response", "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        parsed = json.loads(match.group())
        return str(parsed.get("label", "legitimate")), float(parsed.get("confidence", 0.5)), str(parsed.get("reason", "No reason provided."))
    return "legitimate", 0.5, "Failed to parse reasoning."


def _build_claim_text(extracted_fields: dict) -> str:
    parts = [
        _field_val(extracted_fields, "claimant_name"),
        _field_val(extracted_fields, "policy_number"),
        _field_val(extracted_fields, "incident_type"),
        _field_val(extracted_fields, "amount_claimed"),
        _field_val(extracted_fields, "incident_date"),
        _field_val(extracted_fields, "incident_description"),
    ]
    return " | ".join(p for p in parts if p and p != "N/A")


def _check_duplicate(doc_id: str, claim_text: str) -> tuple[bool, float, str]:
    """Returns (is_duplicate, similarity_score, matching_claim_ref)."""
    model = _get_model()
    emb = model.encode(claim_text).tolist()

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT claim_ref, 1 - (embedding <=> %s::vector) AS similarity
            FROM claims_history
            WHERE claim_ref != %s
            ORDER BY embedding <=> %s::vector
            LIMIT 1;
            """,
            (emb, doc_id, emb),
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row:
            claim_ref, similarity = row
            sim = float(similarity)
            return sim >= DUPLICATE_THRESHOLD, round(sim, 4), str(claim_ref)
    except Exception as e:
        logger.warning("fraud_scoring: duplicate check DB error: %s", e)

    return False, 0.0, ""


def _store_claim_in_history(doc_id: str, extracted_fields: dict, claim_text: str) -> None:
    model = _get_model()
    emb = model.encode(claim_text).tolist()

    def fv(name: str) -> str:
        return _field_val(extracted_fields, name)

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO claims_history (claim_ref, claimant_name, incident_type, amount_claimed, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
            """,
            (doc_id, fv("claimant_name"), fv("incident_type"), fv("amount_claimed"), emb),
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.warning("fraud_scoring: history insert error: %s", e)


def node_fraud_scoring(state: ClaimState) -> dict:
    doc_id = state["doc_id"]
    doc_type = state.get("document_type", "unknown")
    extracted_fields = state.get("extracted_fields", {})
    existing_flags = state.get("flags", [])
    verification = state.get("verification_verdict", {})

    with NodeTracer("fraud_scoring_agent", trace_id=doc_id, inputs={"doc_id": doc_id}) as tracer:
        t0 = time.perf_counter()

        if doc_type != "claim_form":
            result = FraudResult(
                label="legitimate",
                confidence=1.0,
                reason=f"Document classified as '{doc_type}'. Claim fraud evaluation bypassed for non-claim documents.",
                is_duplicate=False,
                duplicate_similarity=0.0,
                duplicate_match_id="",
                method="bypassed_non_claim",
            )
            log = NodeLog(
                node_name="fraud_scoring_agent",
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                langfuse_span_id=tracer.span_id,
                error="",
            )
            tracer.end(outputs={"label": "legitimate", "is_duplicate": False})
            return {
                "fraud_result": result.model_dump(),
                "is_duplicate": False,
                "fraud_score": f"Legitimate — Document is a {doc_type}",
                "flags": [],
                "agent_trace": [log.model_dump()],
            }

        error_msg = ""
        label = "legitimate"
        confidence = 0.5
        reason = "Assessment failed."
        method = "ollama_comprehensive"
        is_dup = False
        dup_sim = 0.0
        dup_ref = ""
        new_flags: list[Flag] = []

        claim_text = _build_claim_text(extracted_fields)

        # Run duplicate check first so the result feeds into the fraud assessment
        try:
            is_dup, dup_sim, dup_ref = _check_duplicate(doc_id, claim_text)
            if is_dup:
                new_flags.append(Flag(
                    code="DUPLICATE_CLAIM",
                    severity="HIGH",
                    description=(
                        f"Claim is {dup_sim:.0%} similar to existing claim '{dup_ref}' "
                        f"in the claims history index (threshold: {DUPLICATE_THRESHOLD:.0%})."
                    ),
                    clause_ref="DUPLICATE_CLAIMS",
                ))
                logger.info("fraud_scoring: DUPLICATE detected similarity=%.3f ref=%s doc_id=%s", dup_sim, dup_ref, doc_id)
        except Exception as e:
            logger.warning("fraud_scoring: duplicate check failed: %s", e)

        try:
            label, confidence, reason = _classify_via_ollama(extracted_fields, existing_flags + [f.model_dump() for f in new_flags], verification, is_dup)
            logger.info("fraud_scoring: label=%s confidence=%.2f method=%s doc_id=%s", label, confidence, method, doc_id)
        except Exception as e:
            logger.warning("fraud_scoring: classification error %s", e)
            error_msg = str(e)
            label, confidence, reason = "legitimate", 0.5, "Error during evaluation."

        if label in ("fraudulent", "suspicious") and confidence > 0.7:
            new_flags.append(Flag(
                code="FRAUD_RISK_HIGH" if label == "fraudulent" else "FRAUD_RISK_MEDIUM",
                severity="HIGH" if label == "fraudulent" else "MEDIUM",
                description=f"Fraud Agent flagged claim as '{label}' ({confidence:.0%} confidence): {reason}",
            ))

        # Store current claim for future duplicate checks
        _store_claim_in_history(doc_id, extracted_fields, claim_text)

        result = FraudResult(
            label=label,
            confidence=round(confidence, 4),
            reason=reason,
            is_duplicate=is_dup,
            duplicate_similarity=dup_sim,
            duplicate_match_id=dup_ref,
            method=method,
        )

        # Legacy string for API compat
        risk_level = "High" if label == "fraudulent" else "Medium" if label == "suspicious" else "Low"
        fraud_score_str = f"{risk_level} risk — {label} (confidence: {confidence:.0%}, method: {method})\nReason: {reason}"

        latency_ms = (time.perf_counter() - t0) * 1000
        log = NodeLog(
            node_name="fraud_scoring_agent",
            latency_ms=round(latency_ms, 1),
            langfuse_span_id=tracer.span_id,
            error=error_msg,
        )
        tracer.end(outputs={"label": label, "is_duplicate": is_dup})

    return {
        "fraud_result": result.model_dump(),
        "is_duplicate": is_dup,
        "fraud_score": fraud_score_str,
        "flags": [f.model_dump() for f in new_flags],
        "agent_trace": [log.model_dump()],
    }
