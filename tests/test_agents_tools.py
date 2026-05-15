from src.agents import tools


def test_tools_return_expected_keys():
    aid = "T-0001"
    assert "snapshot" in tools.get_transformer.invoke({"id": aid})
    assert "fault_type" in tools.get_dga_diagnosis.invoke({"id": aid})
    assert "health_index" in tools.get_health_index.invoke({"id": aid})
    assert "p_failure" in tools.get_failure_probability.invoke({"id": aid, "horizon_years": 5})
    assert "monetized_risk_usd" in tools.get_monetized_risk.invoke({"id": aid, "horizon_years": 5})
    assert "rows" in tools.rank_fleet.invoke({"top_n": 5, "sort_by": "monetized_risk_usd"})
    assert "delta_p5" in tools.simulate_what_if.invoke({"id": aid, "loading_pct": 95, "defer_years": 2, "ambient_delta_c": 2})
    assert "projects" in tools.recommend_replacement_plan.invoke({"budget_musd": 15, "horizon_years": 5})
    assert "recommended_spares" in tools.recommend_spare_strategy.invoke({"voltage_class_kv": 230})
    assert "narrative" in tools.explain_transformer_risk.invoke({"id": aid})
