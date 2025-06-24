import json
from enhanced_analysis_worker import enhanced_analysis_worker


def test_worker_uses_brand_target_variable():
    # Build minimal request mimicking front-end payload for Nike ranking query
    req = {
        "query": "Show me the top 10 areas with highest Nike athletic shoe purchases",
        "analysis_type": "ranking",
        "target_variable": "MP30034A_B",
        "matched_fields": ["MP30034A_B"],
        "metrics": ["MP30034A_B"],
        "conversationContext": "",
        "top_n": 10,
    }

    result = enhanced_analysis_worker(req)
    assert result.get("success"), f"worker error: {result.get('error')}"

    # Validate that worker echoed back correct target_variable in model_info
    assert result["model_info"]["target_variable"].endswith("MP30034A_B"), "target variable not preserved"

    rows = result.get("results", [])
    # Ensure we have rows and at least one non-zero value for Nike metric
    assert len(rows) > 0, "no rows returned"
    nike_key = "value_mp30034a_b" if "value_mp30034a_b" in rows[0] else "mp30034a_b"
    found_non_zero = any(r.get(nike_key) not in (0, 0.0, None) for r in rows)
    assert found_non_zero, "all Nike values are zero – dataset mapping failed" 