from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scanner.board import CharucoTracker


def capture(config: dict, output: str) -> None:
    """Capture RGB-D frames. Press SPACE to save a frame with a valid board pose."""
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "RealSense Python bindings are missing. Run: uv sync --extra hardware"
        ) from exc

    out = Path(output)
    for folder in ("color", "depth", "poses"):
        (out / folder).mkdir(parents=True, exist_ok=True)

    camera = config["camera"]
    pipeline = rs.pipeline()
    rs_config = rs.config()
    if camera.get("serial"):
        rs_config.enable_device(str(camera["serial"]))
    rs_config.enable_stream(
        rs.stream.depth, camera["width"], camera["height"], rs.format.z16, camera["fps"]
    )
    rs_config.enable_stream(
        rs.stream.color, camera["width"], camera["height"], rs.format.bgr8, camera["fps"]
    )

    profile = pipeline.start(rs_config)
    align = rs.align(rs.stream.color)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_profile.get_intrinsics()
    K = np.array(
        [[intrinsics.fx, 0, intrinsics.ppx],
         [0, intrinsics.fy, intrinsics.ppy],
         [0, 0, 1]],
        dtype=np.float64,
    )
    distortion = np.asarray(intrinsics.coeffs[:5], dtype=np.float64)
    tracker = CharucoTracker(config["board"], K, distortion)
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

    metadata = {
        "depth_scale_m_per_unit": depth_scale,
        "intrinsics": {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.ppx,
            "cy": intrinsics.ppy,
        },
        "camera_matrix": K.tolist(),
        "distortion": distortion.tolist(),
        "board": config["board"],
        "board_width_m": config["board"]["squares_x"] * config["board"]["square_length_m"],
        "board_height_m": config["board"]["squares_y"] * config["board"]["square_length_m"],
        "frames": [],
    }

    saved = 0
    print("SPACE = save current view | Q / ESC = finish")
    print("Move the camera slowly. Keep the ChArUco board visible in every saved view.")

    try:
        for _ in range(int(camera.get("warmup_frames", 30))):
            pipeline.wait_for_frames()

        while True:
            frames = align.process(pipeline.wait_for_frames())
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth = np.asanyarray(depth_frame.get_data())
            color = np.asanyarray(color_frame.get_data())
            transform, status, preview = tracker.estimate(color)

            tracking_ok = transform is not None
            color_status = (30, 220, 30) if tracking_ok else (0, 0, 255)
            message = (
                f"TRACKING | corners {status.get('corners', 0)} | "
                f"error {status.get('reprojection_error_px', 0):.2f}px"
                if tracking_ok else f"NO TRACKING | {status.get('reason', 'unknown')}"
            )
            cv2.putText(preview, message, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color_status, 2)
            cv2.putText(preview, f"Saved: {saved} | SPACE save | Q quit",
                        (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("RealSense ChArUco scanner", preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord(" ") and tracking_ok:
                frame_id = f"{saved:05d}"
                cv2.imwrite(str(out / "color" / f"{frame_id}.png"), color)
                cv2.imwrite(str(out / "depth" / f"{frame_id}.png"), depth)
                np.save(out / "poses" / f"{frame_id}.npy", transform)
                metadata["frames"].append({"id": frame_id, "tracking": status})
                saved += 1
                print(f"Saved view {frame_id}: {status}")
            elif key == ord(" ") and not tracking_ok:
                print("Frame not saved: ChArUco pose is invalid.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    metadata["saved_frame_count"] = saved
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {saved} RGB-D views to {out.resolve()}")
