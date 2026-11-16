from pathlib import Path

import yaml


def test_config_matches_physical_charuco_board():
    config = yaml.safe_load(Path("config/scanner.yaml").read_text())
    board = config["board"]

    assert board["squares_x"] == 5
    assert board["squares_y"] == 7
    assert board["square_length_m"] == 0.037
    assert board["marker_length_m"] == 0.0235
    assert board["dictionary"] == "DICT_5X5_50"
    assert board["squares_x"] * board["square_length_m"] == 0.185
    assert board["squares_y"] * board["square_length_m"] == 0.259

def test_visualizer_has_multiple_named_poses():
    config = yaml.safe_load(Path("config/scanner.yaml").read_text())
    poses = config["visualizer"]["poses"]

    assert len(poses) >= 4
    names = [pose["name"] for pose in poses]
    assert len(names) == len(set(names))
    assert {"front", "right", "rear", "left", "top", "isometric"} <= set(names)


def test_decimation_has_low_poly_target():
    config = yaml.safe_load(Path("config/scanner.yaml").read_text())
    assert config["decimation"]["target_triangles"] > 0
    assert config["decimation"]["target_triangles"] < config["reconstruction"]["target_triangles"]
