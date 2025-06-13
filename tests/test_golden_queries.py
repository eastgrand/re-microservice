import json
import os
import sys
import importlib
from pathlib import Path

import pytest

# Allow importing app from parent directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

app_module = importlib.import_module('app')
analysis_worker = app_module.analysis_worker

GOLDEN_PATH = PROJECT_ROOT / 'golden-queries.json'

with open(GOLDEN_PATH, 'r', encoding='utf-8') as f:
    GOLDEN_QUERIES = json.load(f)

@pytest.mark.parametrize('query_obj', GOLDEN_QUERIES)
def test_golden_query(query_obj):
    """Ensure each golden query produces non-empty results and success True."""
    result = analysis_worker(query_obj)
    assert result['success'] is True, "Analysis did not succeed"
    assert isinstance(result['results'], list), "Results not a list"
    assert len(result['results']) > 0, "No results returned"
    assert 'summary' in result and result['summary'], "Summary missing"
