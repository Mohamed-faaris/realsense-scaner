from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from scanner.board import CharucoTracker


def capture(config: dict, output: str, record_video: bool = False) -> None:
    """Capture RGB-D samples; optionally record a continuous MP4 alongside reconstruction data.

    The MP4 is for playback/review. Reconstructable samples are saved as lossless depth PNGs,
    RGB PNGs, camera poses, and metadata in the same session directory.
    """
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
    bag_path = out / "recording.bag" if record_video else None
    if record_video:
        if bag_path.exists() or (out / "recording.mp4").exists():
            raise FileExistsError(
                f"Recording already exists in {out}. Choose a new --output directory."
            )
        # A RealSense bag preserves the raw synchronized depth + color streams.
        rs_config.enable_record_to_file(str(bag_path))
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
        "record_video": bool(record_video),
        "bag_file": bag_path.name if bag_path else None,
        "video_file": None,
        "frames": [],
    }

    saved = 0
    video_writer = None
    video_path = None
    video_frame_index = 0
    last_sample_time = 0.0
    sample_interval = 1.0 / max(
        0.1, float(config.get("capture", {}).get("record_sample_hz", 5.0))
    )

    def save_sample(color: np.ndarray, depth: np.ndarray, transform: np.ndarray,
                    status: dict, timestamp_ms: float, source_video_index: int | None):
        nonlocal saved
        frame_id = f"{saved:06d}"
        if not cv2.imwrite(str(out / "color" / f"{frame_id}.png"), color):
            raise OSError(f"Failed to save RGB frame {frame_id}")
        if not cv2.imwrite(str(out / "depth" / f"{frame_id}.png"), depth):
            raise OSError(f"Failed to save depth frame {frame_id}")
        np.save(out / "poses" / f"{frame_id}.npy", transform)
        metadata["frames"].append({
            "id": frame_id,
            "timestamp_ms": float(timestamp_ms),
            "video_frame_index": source_video_index,
            "tracking": status,
        })
        saved += 1
        print(f"Saved reconstruction sample {frame_id}: {status}")

    print("SPACE = save one valid view | Q / ESC = finish")
    if record_video:
        print("VIDEO RECORDING ENABLED: continuous MP4 + synchronized RGB/depth/pose samples.")
    print("Move slowly and keep the ChArUco board visible in every reconstruction sample.")

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
            timestamp_ms = float(color_frame.get_timestamp())

            if record_video:
                if video_writer is None:
                    height, width = color.shape[:2]
                    video_path = out / "recording.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(video_path), fourcc, float(camera["fps"]), (width, height)
                    )
                    if not video_writer.isOpened():
                        video_writer.release()
                        video_writer = None
                        raise RuntimeError(
                            "Could not open MP4 writer. Check OpenCV FFmpeg/video codec support."
                        )
                # Write every RGB frame for a continuous review video, even if board tracking
                # is temporarily lost. Only valid tracked frames are used for 3D reconstruction.
                video_writer.write(color)
                current_video_index = video_frame_index
                video_frame_index += 1
            else:
                current_video_index = None

            tracking_ok = transform is not None
            color_status = (30, 220, 30) if tracking_ok else (0, 0, 255)
            message = (
                f"TRACKING | corners {status.get('corners', 0)} | "
                f"error {status.get('reprojection_error_px', 0):.2f}px"
                if tracking_ok else f"NO TRACKING | {status.get('reason', 'unknown')}"
            )
            cv2.putText(
                preview, message, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color_status, 2
            )
            hint = "SPACE save | Q quit"
            if record_video:
                hint = "REC | valid views sampled automatically | SPACE force save | Q quit"
            cv2.putText(
                preview, f"Saved: {saved} | {hint}", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2
            )
            cv2.imshow("RealSense ChArUco scanner", preview)
            key = cv2.waitKey(1) & 0xFF

            now = time.monotonic()
            should_auto_save = (
                record_video and tracking_ok and now - last_sample_time >= sample_interval
            )
            if should_auto_save:
                save_sample(color, depth, transform, status, timestamp_ms, current_video_index)
                last_sample_time = now

            if key in (ord("q"), 27):
                break
            if key == ord(" ") and tracking_ok:
                save_sample(color, depth, transform, status, timestamp_ms, current_video_index)
                last_sample_time = now
            elif key == ord(" ") and not tracking_ok:
                print("Frame not saved: ChArUco pose is invalid.")
    finally:
        pipeline.stop()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

    metadata["saved_frame_count"] = saved
    metadata["video_frame_count"] = video_frame_index
    metadata["video_file"] = video_path.name if video_path else None
    metadata["bag_file"] = bag_path.name if bag_path else None
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {saved} reconstructable RGB-D samples to {out.resolve()}")
    if video_path:
        print(f"Continuous review video: {video_path.resolve()}")
    if bag_path:
        print(f"Reprocessable raw RGB-D recording: {bag_path.resolve()}")
