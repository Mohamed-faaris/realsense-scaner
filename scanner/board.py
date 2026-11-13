from __future__ import annotations

import cv2
import numpy as np


def create_board(config: dict):
    """Create the configured ChArUco board using OpenCV's modern ArUco API."""
    aruco = cv2.aruco
    dictionary_name = config["dictionary"]
    dictionary_id = getattr(aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")

    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    board = aruco.CharucoBoard(
        (int(config["squares_x"]), int(config["squares_y"])),
        float(config["square_length_m"]),
        float(config["marker_length_m"]),
        dictionary,
    )
    return dictionary, board


class CharucoTracker:
    """Detect ChArUco automatically and estimate T_camera_board."""

    def __init__(self, config: dict, camera_matrix: np.ndarray, distortion: np.ndarray):
        aruco = cv2.aruco
        self.dictionary, self.board = create_board(config)
        self.detector = aruco.CharucoDetector(self.board)
        self.K = np.asarray(camera_matrix, dtype=np.float64)
        self.dist = np.asarray(distortion, dtype=np.float64)
        self.min_corners = int(config.get("min_corners", 8))
        self.max_error = float(config.get("max_reprojection_error_px", 2.0))

    def estimate(self, bgr: np.ndarray):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, marker_corners, marker_ids = self.detector.detectBoard(gray)

        preview = bgr.copy()
        marker_count = 0 if marker_ids is None else len(marker_ids)
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(preview, marker_corners, marker_ids)

        corner_count = 0 if ids is None else len(ids)
        if corners is None or ids is None or corner_count < self.min_corners:
            return None, {
                "reason": "not enough ChArUco corners",
                "markers": marker_count,
                "corners": corner_count,
            }, preview

        cv2.aruco.drawDetectedCornersCharuco(preview, corners, ids)
        object_points = self.board.getChessboardCorners()[
            ids.flatten().astype(int)
        ].astype(np.float64)
        image_points = corners.reshape(-1, 2).astype(np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.K,
            self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None, {
                "reason": "solvePnP failed",
                "markers": marker_count,
                "corners": corner_count,
            }, preview

        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, self.K, self.dist
        )
        projected = projected.reshape(-1, 2)
        error = float(
            np.sqrt(np.mean(np.sum((image_points - projected) ** 2, axis=1)))
        )
        status = {
            "markers": marker_count,
            "corners": corner_count,
            "reprojection_error_px": error,
        }
        if error > self.max_error:
            status["reason"] = "reprojection error too high"
            return None, status, preview

        rotation, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = tvec.reshape(3)

        cv2.drawFrameAxes(preview, self.K, self.dist, rvec, tvec, 0.05)
        status["reason"] = "tracking"
        return transform, status, preview


def save_board_image(config: dict, output: str, pixels_per_square: int = 400) -> None:
    """Generate the configured board image without adding a print margin."""
    _, board = create_board(config)
    width = int(config["squares_x"]) * pixels_per_square
    height = int(config["squares_y"]) * pixels_per_square
    image = board.generateImage((width, height), marginSize=0, borderBits=1)
    if not cv2.imwrite(output, image):
        raise OSError(f"Could not write board image: {output}")
    width_mm = config["squares_x"] * config["square_length_m"] * 1000
    height_mm = config["squares_y"] * config["square_length_m"] * 1000
    print(f"Saved {output}. Print the board at {width_mm:.0f} x {height_mm:.0f} mm.")
