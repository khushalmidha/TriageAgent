"""
Automated unit tests for Golden Ground-Truth Protocol-ID Resolution and Schema Integrity.
Prevents regression of the protocol-ID alignment bug where generated KB protocol IDs
diverged from expected test case identifiers.
"""

import json
import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

@pytest.fixture(scope="module")
def golden_cases():
    golden_path = ROOT_DIR / "data" / "golden_test_set.json"
    assert golden_path.exists(), f"Golden test set missing at {golden_path}"
    with open(golden_path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def kb_records():
    kb_path = ROOT_DIR / "data" / "knowledge_base" / "triage_knowledge_base.json"
    assert kb_path.exists(), f"Knowledge base missing at {kb_path}"
    with open(kb_path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def chunks_metadata():
    chunks_path = ROOT_DIR / "data" / "triage_chunks.json"
    assert chunks_path.exists(), f"Chunks metadata missing at {chunks_path}"
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_golden_test_set_schema(golden_cases):
    """Verifies that all 100 golden cases adhere to the required schema."""
    assert len(golden_cases) == 100, f"Expected 100 golden cases, found {len(golden_cases)}"
    
    valid_severities = {"Low", "Medium", "High", "Critical"}
    valid_categories = {"clinical_symptom", "transaction_dispute", "account_security"}

    for idx, case in enumerate(golden_cases):
        assert "query" in case and len(case["query"].strip()) > 5, f"Case {idx} invalid query"
        assert "expected_severity" in case and case["expected_severity"] in valid_severities, f"Case {idx} invalid severity"
        assert "expected_category" in case and case["expected_category"] in valid_categories, f"Case {idx} invalid category"
        assert "canonical_guideline_title" in case, f"Case {idx} missing canonical_guideline_title"
        assert "expected_reference_chunks" in case and len(case["expected_reference_chunks"]) > 0, f"Case {idx} missing chunks"

def test_golden_protocol_ids_resolve(golden_cases, kb_records):
    """
    REGRESSION TEST: Ensures all canonical guideline titles in the golden test set
    map to at least one valid protocol in the knowledge base.
    """
    # Extract all canonical guideline base titles from KB
    kb_canonical_titles = set()
    for item in kb_records:
        raw_title = (
            item["title"]
            .split(" - Clinical Guideline Variant")[0]
            .split(" - Security Protocol Variant")[0]
            .split(" - Financial Protocol Variant")[0]
            .split(" - SLA Variant")[0]
            .split(" - Dispute Matrix Variant")[0]
            .split(" - SLA Policy Variant")[0]
            .strip()
        )
        kb_canonical_titles.add(raw_title)

    unmapped = []
    for case in golden_cases:
        canon_title = case.get("canonical_guideline_title", "").strip()
        if canon_title not in kb_canonical_titles:
            unmapped.append((case.get("case_id", "UNK"), canon_title))

    assert len(unmapped) == 0, f"Found {len(unmapped)} unmapped canonical guideline titles: {unmapped[:5]}"

def test_retrieved_ids_match_canonical_ids(golden_cases, chunks_metadata):
    """
    REGRESSION TEST: Ensures all expected reference chunks resolve to valid chunk IDs or protocol IDs.
    """
    known_protocol_ids = {c["protocol_id"].upper() for c in chunks_metadata}
    known_chunk_ids = {c.get("chunk_id", "").upper() for c in chunks_metadata}
    known_all = known_protocol_ids | known_chunk_ids

    unresolvable = []
    for case in golden_cases:
        for exp in case.get("expected_reference_chunks", []):
            base_pid = exp.split("-CHUNK-")[0].strip().upper()
            if base_pid not in known_protocol_ids and exp.upper() not in known_all:
                unresolvable.append((case.get("case_id", "UNK"), exp))

    assert len(unresolvable) == 0, f"Found {len(unresolvable)} unresolvable reference chunk IDs: {unresolvable[:5]}"
