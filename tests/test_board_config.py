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
