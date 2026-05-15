from src.models.dga_diagnostics import duval_triangle_1


def test_duval_triangle_known_points():
    assert duval_triangle_1(10, 80, 30) == "D2"
    assert duval_triangle_1(70, 5, 15) in {"PD", "DT", "T1"}
    assert duval_triangle_1(10, 2, 80) in {"T2", "T3"}
