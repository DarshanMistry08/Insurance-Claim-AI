"""
api/schemas.py — Pydantic models for the FastAPI request/response layer.
Extended to include agent_trace and structured sub-objects.
"""

from pydantic import BaseModel
from typing import Optional, Any

class ClaimResult(BaseModel):
    doc_id: str

    # Document Classification
    document_type: str = "unknown"
    document_type_confidence: float = 0.0
    classification_status: str = "certain"

    # Core extracted fields (legacy flat dict for UI compatibility)
    extracted_fields: dict[str, Any]        # {field_name: value_string or dict}
    policy_context: str
    is_duplicate: bool
    fraud_score: str
    flags: list[Any]                        # flat flag strings or dicts
    final_summary: Any

    # Extended structured fields
    agent_trace: list[dict] = []            # [NodeLog.model_dump()]
    verification_verdict: dict = {}         # VerificationVerdict.model_dump()
    fraud_result: dict = {}                 # FraudResult.model_dump()
    recommendation: str = ""               # APPROVE | FLAG_FOR_REVIEW | REJECT
    confidence_score: float = 0.0
    citations: list[dict] = []             # [Citation.model_dump()]
    retrieved_clauses: list[dict] = []     # [PolicyClause.model_dump()]


class HealthResponse(BaseModel):
    status: str
    ollama: str
    postgres: str
    hf_token_set: bool = False
    langfuse_set: bool = False
