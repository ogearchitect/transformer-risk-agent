from src.models.failure_prob import get_asset_probability


def test_failure_probability_bounds():
    p1 = get_asset_probability("T-0001", 1)
    p5 = get_asset_probability("T-0001", 5)
    assert 0 <= p1 <= 1
    assert 0 <= p5 <= 1
