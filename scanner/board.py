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
    """Detect the board and estimate T_camera_board for each color frame."""

    def __init__(self, config: dict, camera_matrix: np.ndarray, distortion: np.ndarray):
        aruco = cv2.aruco
        self.dictionary, self.board = create_board(config)
        self.detector = aruco.ArucoDetector(
            self.dictionary,
            aruco.DetectorParameters(),
        )
        self.K = np.asarray(camera_matrix, dtype=np.float64)
        self.dist = np.asarray(distortion, dtype=np.float64)
        self.min_corners = int(config.get("min_corners", 8))
        self.max_error = float(config.get("max_reprojection_error_px", 2.0))

    def estimate(self, bgr: np.ndarray):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = self.detector.detectMarkers(gray)

        if marker_ids is None:
            return None, {"reason": "no markers", "markers": 0, "corners": 0}, bgr

        count, corners, ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            self.board,
            cameraMatrix=self.K,
            distCoeffs=self.dist,
        )

        preview = bgr.copy()
        cv2.aruco.drawDetectedMarkers(preview, marker_corners, marker_ids)

        if corners is None or int(count) < self.min_corners:
            return None, {
                "reason": "not enough ChArUco corners",
                "markers": len(marker_ids),
                "corners": int(count or 0),
            }, preview

        cv2.aruco.drawDetectedCornersCharuco(preview, corners, ids)
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            corners, ids, self.board, self.K, self.dist, None, None
        )
        if not ok:
            return None, {
                "reason": "pose estimation failed",
                "markers": len(marker_ids),
                "corners": int(count),
            }, preview

        board_corners = self.board.getChessboardCorners()
        object_points = board_corners[ids.flatten().astype(int)].astype(np.float64)
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, self.K, self.dist)
        observed = corners.reshape(-1, 2)
        projected = projected.reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum((observed - projected) ** 2, axis=1))))

        status = {
            "markers": len(marker_ids),
            "corners": int(count),
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
    """Generate a board image with a 5:7 layout ratio; print at measured 1:1 size."""
    _, board = create_board(config)
    width = int(config["squares_x"]) * pixels_per_square
    height = int(config["squares_y"]) * pixels_per_square
    image = board.generateImage((width, height), marginSize=0, borderBits=1)
    if not cv2.imwrite(output, image):
        raise OSError(f"Could not write board image: {output}")
    print(
        f"Saved {output}. Print the chessboard area at "
        f"{config['squares_x'] * config['square_length_m'] * 1000:.0f} x "
        f"{config['squares_y'] * config['square_length_m'] * 1000:.0f} mm."
    )
