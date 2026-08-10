from __future__ import annotations
import os
import re
import json
import time
import logging
import requests

from agents.state import (
    ClaimState, VerificationVerdict, VerificationFinding, NodeLog, Flag
)
from agents.tracing import NodeTracer

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")


def _field_val(extracted_fields: dict, name: str) -> str:
    f = extracted_fields.get(name, {})
    if isinstance(f, dict):
        v = f.get("value", "")
        return v if v else ""
    return str(f) if f else ""


def _handle_policy_schedule(
    doc_id: str,
    extracted_fields: dict,
    tracer,
    t0: float,
) -> dict:
    """
    Builds the verification output directly from extracted schedule fields.
    No LLM call needed — there's no incident to verify against policy clauses.
    """
    sum_insured        = _field_val(extracted_fields, "sum_insured")
    scheme_description = _field_val(extracted_fields, "scheme_description")
    policy_type        = _field_val(extracted_fields, "policy_type")
    period_from        = _field_val(extracted_fields, "period_of_insurance_from")
    period_to          = _field_val(extracted_fields, "period_of_insurance_to")
    proposer_name      = _field_val(extracted_fields, "proposer_name")
    policy_number      = _field_val(extracted_fields, "policy_number")
    insured_persons    = _field_val(extracted_fields, "insured_persons")

    covered_items_parts = []
    if scheme_description:
        covered_items_parts.append(scheme_description)
    if policy_type and policy_type.lower() not in (scheme_description or "").lower():
        covered_items_parts.append(policy_type)
    if insured_persons:
        covered_items_parts.append(f"Insured: {insured_persons}")
    covered_items = "; ".join(covered_items_parts) if covered_items_parts else "Not applicable to policy schedule"

    coverage_limit = sum_insured if sum_insured else "Not applicable to policy schedule"

    if period_from and period_to:
        policy_period = f"{period_from} to {period_to}"
    elif period_from:
        policy_period = f"From {period_from}"
    else:
        policy_period = "Not stated in schedule"

    # Restrictions, exclusions, and deductibles live in the policy wording document, not the schedule
    na_msg = "Not applicable to policy schedule"

    key_fields_ok = bool(sum_insured and scheme_description)
    match_status = "Schedule Verified" if key_fields_ok else "Uncertain — Key Fields Missing"

    # Build structured policy_context in the exact format parsePolicyContext() on the frontend expects
    policy_context = (
        f"RELEVANT POLICY FINDINGS\n\n"
        f"Covered Items:\n- {covered_items}\n\n"
        f"Coverage Limit:\n- {coverage_limit}\n\n"
        f"Restrictions:\n- {na_msg}\n\n"
        f"Exclusions:\n- {na_msg}\n\n"
        f"Deductibles:\n- {na_msg}\n\n"
        f"Policy Match Status:\n- {match_status}"
    )

    verdict = VerificationVerdict(
        coverage_status="schedule_verified" if key_fields_ok else "uncertain",
        amount_within_limit=True,
        policy_period_valid=True,
        findings=[
            VerificationFinding(
                field_ref="sum_insured",
                clause_ref="POLICY_SCHEDULE",
                result="within_limit",
                evidence=f"Sum Insured extracted from schedule: {coverage_limit}",
            ),
            VerificationFinding(
                field_ref="scheme_description",
                clause_ref="POLICY_SCHEDULE",
                result="covered",
                evidence=f"Scheme: {covered_items}",
            ),
        ] if key_fields_ok else [],
    )

    log = NodeLog(
        node_name="verification_agent",
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        langfuse_span_id=tracer.span_id,
        error="",
    )
    tracer.end(outputs={
        "coverage_status": verdict.coverage_status,
        "findings": len(verdict.findings),
    })

    return {
        "verification_verdict": verdict.model_dump(),
        "policy_context": policy_context,
        "flags": [],
        "agent_trace": [log.model_dump()],
    }


def _build_verification_prompt(extracted_fields: dict, retrieved_clauses: list) -> str:
    def field_val(name: str) -> str:
        f = extracted_fields.get(name, {})
        return (f.get("value", "N/A") if isinstance(f, dict) else str(f))

    fields_text = "\n".join(
        f"  {name}: {field_val(name)}" for name in [
            "claimant_name", "policy_number", "incident_date", "claim_filing_date",
            "incident_type", "incident_description", "amount_claimed",
        ]
    )

    clauses_text = "\n\n".join(
        f"[{c.get('clause_id', 'CLAUSE')}] ({c.get('policy_doc', 'unknown')} — similarity {c.get('similarity_score', 0):.2f})\n{c.get('chunk_text', '')}"
        for c in retrieved_clauses
    ) if retrieved_clauses else "No policy clauses retrieved."

    return f"""You are a senior insurance claims adjuster AI. Your task is to verify a claim against the provided policy text.

CLAIM FIELDS:
{fields_text}

RETRIEVED POLICY CLAUSES:
{clauses_text}

INSTRUCTIONS:
- Compare each claim field against the relevant policy clause.
- Determine if the claim is covered, excluded, or uncertain.
- Check: (1) incident type is a covered peril, (2) amount is within the clause limit, (3) incident date falls within the policy period.
- For each finding, cite the exact clause_id and field_ref.
- Do NOT invent values not present in the claim fields or clauses.

Respond with ONLY this JSON object (no markdown, no explanation):
{{
  "coverage_status": "covered" | "excluded" | "uncertain",
  "amount_within_limit": true | false,
  "policy_period_valid": true | false,
  "findings": [
    {{
      "field_ref": "<field name>",
      "clause_ref": "<CLAUSE_ID>",
      "result": "within_limit" | "exceeds_limit" | "excluded" | "covered" | "uncertain",
      "evidence": "<exact quoted text from policy clause that supports this finding>"
    }}
  ]
}}"""


def _call_ollama(prompt: str) -> dict:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=90,
    )
    resp.raise_for_status()
    raw = resp.json().get("response", "").strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {raw[:200]}")
    return json.loads(match.group())


def _handle_claim_form(
    doc_id: str,
    extracted_fields: dict,
    retrieved_clauses: list,
    state: ClaimState,
    tracer,
    t0: float,
) -> dict:
    error_msg = ""
    verdict = VerificationVerdict()
    new_flags: list[Flag] = []

    try:
        prompt = _build_verification_prompt(extracted_fields, retrieved_clauses)
        parsed = _call_ollama(prompt)

        findings = []
        for f in parsed.get("findings", []):
            try:
                findings.append(VerificationFinding(
                    field_ref=f.get("field_ref", ""),
                    clause_ref=f.get("clause_ref", ""),
                    result=f.get("result", "uncertain"),
                    evidence=f.get("evidence", ""),
                ))
            except Exception:
                pass

        verdict = VerificationVerdict(
            coverage_status=parsed.get("coverage_status", "uncertain"),
            amount_within_limit=bool(parsed.get("amount_within_limit", True)),
            policy_period_valid=bool(parsed.get("policy_period_valid", True)),
            findings=findings,
        )

        # Raise flags from verdict
        if verdict.coverage_status == "excluded":
            new_flags.append(Flag(
                code="COVERAGE_EXCLUDED",
                severity="HIGH",
                description="Policy verification determined this incident type is explicitly excluded.",
                clause_ref=findings[0].clause_ref if findings else "",
            ))
        if not verdict.policy_period_valid:
            new_flags.append(Flag(
                code="OUTSIDE_POLICY_PERIOD",
                severity="HIGH",
                description="Incident date falls outside the active policy period.",
                field_ref="incident_date",
                clause_ref="POLICY_PERIOD",
            ))
        if not verdict.amount_within_limit:
            existing_codes = {f.get("code", "") for f in state.get("flags", [])}
            if not any("EXCEEDS" in c for c in existing_codes):
                new_flags.append(Flag(
                    code="AMOUNT_EXCEEDS_POLICY_LIMIT",
                    severity="HIGH",
                    description="LLM verification determined claimed amount exceeds the applicable policy limit.",
                    field_ref="amount_claimed",
                ))

        logger.info(
            "verification: status=%s amount_ok=%s period_ok=%s doc_id=%s",
            verdict.coverage_status, verdict.amount_within_limit, verdict.policy_period_valid, doc_id,
        )

    except Exception as e:
        error_msg = str(e)
        logger.error("verification: failed doc_id=%s error=%s", doc_id, e)

    latency_ms = (time.perf_counter() - t0) * 1000
    log = NodeLog(
        node_name="verification_agent",
        latency_ms=round(latency_ms, 1),
        langfuse_span_id=tracer.span_id,
        error=error_msg,
    )
    tracer.end(outputs={
        "coverage_status": verdict.coverage_status,
        "findings": len(verdict.findings),
    })

    return {
        "verification_verdict": verdict.model_dump(),
        "flags": [f.model_dump() for f in new_flags],
        "agent_trace": [log.model_dump()],
    }


def node_verification(state: ClaimState) -> dict:
    doc_id = state["doc_id"]
    doc_type = state.get("document_type", "unknown")
    extracted_fields = state.get("extracted_fields", {})
    retrieved_clauses = state.get("retrieved_clauses", [])

    with NodeTracer(
        "verification_agent",
        trace_id=doc_id,
        inputs={"doc_id": doc_id, "doc_type": doc_type, "num_clauses": len(retrieved_clauses)},
    ) as tracer:
        t0 = time.perf_counter()

        if doc_type == "policy_schedule":
            return _handle_policy_schedule(doc_id, extracted_fields, tracer, t0)

        elif doc_type == "claim_form":
            return _handle_claim_form(doc_id, extracted_fields, retrieved_clauses, state, tracer, t0)

        else:
            verdict = VerificationVerdict(
                coverage_status="not_applicable",
                amount_within_limit=True,
                policy_period_valid=True,
                findings=[],
            )
            log = NodeLog(
                node_name="verification_agent",
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                langfuse_span_id=tracer.span_id,
                error="",
            )
            tracer.end(outputs={"coverage_status": "not_applicable", "findings": 0})
            return {
                "verification_verdict": verdict.model_dump(),
                "flags": [],
                "agent_trace": [log.model_dump()],
            }
