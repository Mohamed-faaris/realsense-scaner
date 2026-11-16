from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scanner.board import CharucoTracker


def replay_bag(bag_path: str, output: str, config: dict) -> None:
    """Re-run ChArUco detection over a recorded RealSense .bag without camera hardware."""
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "RealSense Python bindings are missing. Run: uv sync --extra hardware"
        ) from exc

    source = Path(bag_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    out = Path(output)
    for folder in ("color", "depth", "poses"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    if (out / "metadata.json").exists():
        raise FileExistsError(
            f"{out / 'metadata.json'} already exists. Choose a fresh output directory."
        )

    pipeline = rs.pipeline()
    playback_config = rs.config()
    playback_config.enable_device_from_file(str(source), repeat_playback=False)
    profile = pipeline.start(playback_config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
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

    sample_hz = max(0.1, float(config.get("capture", {}).get("record_sample_hz", 5.0)))
    sample_interval_ms = 1000.0 / sample_hz
    last_saved_timestamp = None
    saved = 0
    decoded_frames = 0

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
        "record_video": False,
        "source_bag": str(source),
        "frames": [],
    }

    print(f"Replaying {source}")
    print(f"Re-detecting ChArUco poses; sampling up to {sample_hz:g} valid views/sec.")

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                # Non-real-time playback signals end-of-file by stopping the playback device.
                if playback.current_status() == rs.playback_status.stopped:
                    break
                raise

            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            decoded_frames += 1
            timestamp_ms = float(color_frame.get_timestamp())
            if (
                last_saved_timestamp is not None
                and timestamp_ms - last_saved_timestamp < sample_interval_ms
            ):
                continue

            depth = np.asanyarray(depth_frame.get_data())
            color = np.asanyarray(color_frame.get_data())
            transform, status, _ = tracker.estimate(color)
            if transform is None:
                continue

            frame_id = f"{saved:06d}"
            if not cv2.imwrite(str(out / "color" / f"{frame_id}.png"), color):
                raise OSError(f"Failed to save replay RGB frame {frame_id}")
            if not cv2.imwrite(str(out / "depth" / f"{frame_id}.png"), depth):
                raise OSError(f"Failed to save replay depth frame {frame_id}")
            np.save(out / "poses" / f"{frame_id}.npy", transform)
            metadata["frames"].append({
                "id": frame_id,
                "timestamp_ms": timestamp_ms,
                "source_frame_index": decoded_frames - 1,
                "tracking": status,
            })
            saved += 1
            last_saved_timestamp = timestamp_ms
            print(
                f"Accepted {frame_id}: {status['corners']} corners, "
                f"reprojection {status['reprojection_error_px']:.2f}px"
            )
    finally:
        pipeline.stop()

    metadata["saved_frame_count"] = saved
    metadata["decoded_frame_count"] = decoded_frames
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Decoded {decoded_frames} frames; saved {saved} valid RGB-D poses.")
    print(f"Replay dataset: {out.resolve()}")
    if saved == 0:
        print("No frames passed ChArUco tracking. Check board visibility and board config.")


