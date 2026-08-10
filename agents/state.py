from __future__ import annotations
from typing import TypedDict, Annotated
import operator
from pydantic import BaseModel, Field


class ExtractedField(BaseModel):
    value: str = ""
    confidence: float = 0.0
    status: str = "EXTRACTED"  # "EXTRACTED" | "NOT_FOUND" | "EXTRACTION_FAILED"
    missing_reason: str = ""
    extraction_method: str = "unknown"

    def to_str(self) -> str:
        return self.value


class PolicyClause(BaseModel):
    clause_id: str = ""
    policy_doc: str = ""
    chunk_text: str = ""
    similarity_score: float = 0.0


class VerificationFinding(BaseModel):
    field_ref: str            # e.g. "amount_claimed"
    clause_ref: str           # e.g. "WATER_DAMAGE_B"
    result: str               # "within_limit" | "exceeds_limit" | "excluded" | "uncertain"
    evidence: str             # quoted snippet from policy clause


class VerificationVerdict(BaseModel):
    coverage_status: str = "uncertain"   # "covered" | "excluded" | "uncertain"
    amount_within_limit: bool = True
    policy_period_valid: bool = True
    findings: list[VerificationFinding] = Field(default_factory=list)


class FraudResult(BaseModel):
    label: str = "legitimate"       # "legitimate" | "fraudulent" | "suspicious"
    confidence: float = 0.0
    reason: str = ""
    is_duplicate: bool = False
    duplicate_similarity: float = 0.0
    duplicate_match_id: str = ""
    method: str = "unknown"         # "bart_zero_shot" | "ollama_fallback"


class Flag(BaseModel):
    code: str                        # e.g. "AMOUNT_EXCEEDS_WATER_LIMIT"
    severity: str = "MEDIUM"         # "HIGH" | "MEDIUM" | "LOW"
    description: str
    field_ref: str = ""
    clause_ref: str = ""

    def to_str(self) -> str:
        return f"{self.code}: {self.description}"


class Citation(BaseModel):
    field_ref: str = ""
    clause_ref: str = ""
    rationale: str


class AdjusterSummary(BaseModel):
    recommendation: str = "FLAG_FOR_REVIEW"   # "APPROVE" | "FLAG_FOR_REVIEW" | "REJECT"
    confidence_score: float = 0.0
    summary_text: str = ""
    citations: list[Citation] = Field(default_factory=list)


class NodeLog(BaseModel):
    node_name: str
    latency_ms: float = 0.0
    langfuse_span_id: str = ""
    extraction_method: str = ""
    error: str = ""


class ClaimState(TypedDict):
    # Input
    doc_id: str
    image_path: str

    # Document Classification
    document_type: str            # policy_schedule | claim_form | medical_bill_invoice | discharge_summary | kyc_id_proof | claims_history_record | unknown
    document_type_confidence: float
    classification_status: str    # "certain" | "uncertain"

    # Agent outputs (plain dicts — JSON-serialisable)
    extracted_fields: dict        # {field_name: ExtractedField.model_dump()}
    retrieved_clauses: list       # [PolicyClause.model_dump(), ...]
    verification_verdict: dict    # VerificationVerdict.model_dump()
    fraud_result: dict            # FraudResult.model_dump()
    final_summary: dict           # AdjusterSummary.model_dump()

    # Accumulated lists — each node appends, LangGraph merges
    flags: Annotated[list, operator.add]          # [Flag.model_dump(), ...]
    agent_trace: Annotated[list, operator.add]    # [NodeLog.model_dump(), ...]

    # Flattened legacy fields (kept for API/eval backward-compat)
    policy_context: str
    is_duplicate: bool
    fraud_score: str
    final_summary_text: str


def empty_state(doc_id: str, image_path: str) -> ClaimState:
    return ClaimState(
        doc_id=doc_id,
        image_path=image_path,
        document_type="unknown",
        document_type_confidence=0.0,
        classification_status="uncertain",
        extracted_fields={},
        retrieved_clauses=[],
        verification_verdict={},
        fraud_result={},
        final_summary={},
        flags=[],
        agent_trace=[],
        policy_context="",
        is_duplicate=False,
        fraud_score="",
        final_summary_text="",
    )
