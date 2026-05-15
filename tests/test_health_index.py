from src.data.loader import load_seed_data
from src.models.health_index import compute_health_index


def test_health_index_bounds():
    data = load_seed_data()
    aid = data["transformers"]["id"].iloc[0]
    age = 2026 - int(data["transformers"].set_index("id").loc[aid]["vintage_year"])
    result = compute_health_index(
        data["dga_history"][data["dga_history"]["id"] == aid],
        data["pd_history"][data["pd_history"]["id"] == aid],
        data["bushing_history"][data["bushing_history"]["id"] == aid],
        data["maintenance_log"][data["maintenance_log"]["id"] == aid],
        age,
    )
    assert 0 <= result["health_index"] <= 100
