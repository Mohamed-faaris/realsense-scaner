from pathlib import Path

import cv2
import numpy as np
import yaml

from scanner.diagnostics import remove_depth_outliers


def load_config():
    return yaml.safe_load(Path("config/scanner.yaml").read_text())


def test_config_matches_physical_charuco_board():
    board = load_config()["board"]

    assert board["squares_x"] == 5
    assert board["squares_y"] == 7
    assert board["square_length_m"] == 0.037
    assert board["marker_length_m"] == 0.0235
    assert board["dictionary"] == "DICT_5X5_50"
    assert board["squares_x"] * board["square_length_m"] == 0.185
    assert board["squares_y"] * board["square_length_m"] == 0.259


def test_visualizer_has_multiple_named_poses():
    poses = load_config()["visualizer"]["poses"]
    names = [pose["name"] for pose in poses]

    assert len(poses) >= 4
    assert len(names) == len(set(names))
    assert {"front", "right", "rear", "left", "top", "isometric"} <= set(names)


def test_outlier_removal_is_enabled_and_depth_range_is_configured():
    config = load_config()

    assert config["outlier_removal"]["enabled"] is True
    assert config["outlier_removal"]["median"]["enabled"] is True
    assert config["camera"]["min_depth_m"] > 0
    assert config["camera"]["max_depth_m"] > config["camera"]["min_depth_m"]


def test_median_outlier_filter_rejects_isolated_depth_spike():
    depth = np.full((9, 9), 600, dtype=np.uint16)
    depth[4, 4] = 1200
    config = {
        "outlier_removal": {
            "enabled": True,
            "median": {
                "enabled": True,
                "kernel_size": 3,
                "max_deviation_mm": 20,
            },
            "statistical": {"enabled": False},
        }
    }
    intrinsics = {"fx": 600.0, "fy": 600.0, "cx": 4.0, "cy": 4.0}

    filtered, stats = remove_depth_outliers(
        depth, 0.001, intrinsics, config
    )

    assert filtered[4, 4] == 0
    assert np.count_nonzero(filtered) == 80
    assert stats["median_rejected"] == 1
    assert stats["output_pixels"] == 80


def test_outlier_filter_can_be_disabled():
    depth = np.full((5, 5), 600, dtype=np.uint16)
    depth[2, 2] = 1200
    filtered, stats = remove_depth_outliers(
        depth,
        0.001,
        {"fx": 600.0, "fy": 600.0, "cx": 2.0, "cy": 2.0},
        {"outlier_removal": {"enabled": False}},
    )

    assert np.array_equal(filtered, depth)
    assert stats["enabled"] is False


def test_decimation_and_mesh_cleanup_are_configured():
    config = load_config()

    assert config["decimation"]["target_triangles"] > 0
    assert config["mesh_cleanup"]["remove_small_components"] is True
    assert config["mesh_cleanup"]["min_component_triangles"] > 0
