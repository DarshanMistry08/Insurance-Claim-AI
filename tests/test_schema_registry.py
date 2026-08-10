"""
tests/test_schema_registry.py — Automated Registry Completeness Test.

Enforces:
  1. Every document_type in DOCUMENT_TYPES has a schema in SCHEMA_REGISTRY.
  2. Every schema field has a non-empty extraction prompt in EXTRACTION_PROMPT_REGISTRY.
  3. Fails the build if any schema field lacks a registered extraction prompt.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.schema_registry import (
    DOCUMENT_TYPES,
    SCHEMA_REGISTRY,
    EXTRACTION_PROMPT_REGISTRY,
    get_prompt_for_field,
)


def test_schema_registry_completeness():
    print("Running Automated Schema & Prompt Registry Completeness Test...")
    missing_prompts = []
    missing_schemas = []

    for doc_type in DOCUMENT_TYPES:
        if doc_type == "unknown":
            continue

        if doc_type not in SCHEMA_REGISTRY:
            missing_schemas.append(doc_type)
            continue

        schema_fields = SCHEMA_REGISTRY[doc_type]
        for field_name in schema_fields.keys():
            prompt = get_prompt_for_field(doc_type, field_name)
            if not prompt or not prompt.strip():
                missing_prompts.append(f"{doc_type}.{field_name}")

    if missing_schemas:
        raise AssertionError(f"BUILD FAILED: Missing schemas for document types: {missing_schemas}")

    if missing_prompts:
        raise AssertionError(f"BUILD FAILED: The following schema fields lack registered extraction prompts: {missing_prompts}")

    print("PASS: All document types and schema fields have non-empty registered extraction prompts!")


if __name__ == "__main__":
    test_schema_registry_completeness()
