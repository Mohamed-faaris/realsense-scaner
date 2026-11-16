# RealSense Scaner

A small D415 object-scanning tool. Move the camera around an object placed on a ChArUco board. OpenCV detects the board automatically and estimates camera pose. Open3D fuses RGB-D views into an approximate low-poly mesh.

Python-first: CLI, YAML config, OpenCV preview and Open3D visualization. No web frontend or database.

## Stack

- Python 3.11 + `uv`
- RealSense SDK / `pyrealsense2`
- OpenCV contrib (automatic ChArUco detection)
- Open3D (TSDF reconstruction, mesh simplification, rendering)
- YAML configuration

## Board

Configured for the existing board: 5 × 7 squares, 37 mm squares, 23.5 mm markers, 185 × 259 mm overall, `DICT_5X5_50`, 17 ArUco markers and 24 ChArUco inner corners.

Generate the exact configured board image:

```bash
uv run rs-scan board --output charuco-board.png
```

Print at actual size and measure the physical squares.

## Install

Install the Intel RealSense SDK and udev rules for Ubuntu first:
https://github.com/realsenseai/librealsense

```bash
git clone https://github.com/Mohamed-faaris/realsense-scaner.git
cd realsense-scaner
uv python install 3.11
uv sync --extra hardware
```

If `pyrealsense2` has no compatible wheel for your Python/Ubuntu version, install its binding using the librealsense instructions and run `uv sync`.

## Capture

Manual viewpoint capture:

```bash
uv run rs-scan capture --output datasets/scan01
```

- Move the camera slowly.
- Keep at least 8 ChArUco corners visible.
- Press SPACE to save a valid RGB-D view.
- Press Q or ESC to finish.

Continuous recording mode:

```bash
uv run rs-scan capture --output datasets/scan01 --record-video
```

This saves a complete session directory:

- `recording.bag`: raw synchronized RealSense color + depth streams, timestamps and device stream metadata. This is the **reprocessable source recording**.
- `recording.mp4`: continuous RGB playback video for quick review.
- `color/*.png`: RGB frames selected for reconstruction.
- `depth/*.png`: aligned 16-bit depth frames for the same samples.
- `poses/*.npy`: ChArUco camera pose for each selected sample.
- `metadata.json`: intrinsics, depth scale, board settings, timestamps and sample mapping.

Valid RGB-D samples are saved automatically at `capture.record_sample_hz` (default 5 Hz). SPACE forces an extra sample. MP4 continues recording even when tracking is temporarily lost; invalid-pose frames are not accepted for 3D fusion.

**Do not treat MP4 as the 3D source.** RGB video does not contain metric depth. The RealSense `.bag` preserves the raw depth and color streams required to regenerate reconstruction samples later.

## Reprocess a recording later (camera not required)

You can re-run board detection using the saved bag file. This is useful if you change the board detector, sampling frequency or pose filtering:

```bash
uv run rs-scan replay-bag \
  --bag datasets/scan01/recording.bag \
  --output datasets/scan01_reprocessed
```

This replays the recorded depth and color streams, detects ChArUco poses again, and writes a normal RGB-D dataset.

Reconstruct from either the original capture dataset or the replayed dataset:

```bash
uv run rs-scan reconstruct \
  --dataset datasets/scan01_reprocessed \
  --output outputs/scan01
```

No physical D415 is needed for replay or reconstruction. You only need the saved `.bag` file and the project environment. Librealsense supports recording to and playback from bag files through `enable_record_to_file` and `enable_device_from_file`.

## Render multiple viewpoints into one image

Configure the camera poses in `visualizer.poses` inside `config/scanner.yaml`. Each entry has a name, azimuth, elevation and distance multiplier.

```bash
uv run rs-scan visualize \
  --mesh outputs/scan01/mesh.ply \
  --output outputs/scan01/views
```

Creates individual angle images and one combined `all_views.png` contact sheet with every configured view.

Default views: front, right, rear, left, top and isometric. Rendering uses Open3D OffscreenRenderer; Linux needs a working OpenGL/EGL graphics runtime.

## Decimate a mesh

```bash
uv run rs-scan decimate \
  --input outputs/scan01/mesh.ply \
  --output outputs/scan01/mesh_decimated.ply \
  --target-triangles 1500
```

Without `--target-triangles`, the command uses `decimation.target_triangles` from YAML. Quadric-error decimation writes the simplified mesh and a JSON report with before/after vertex and triangle counts. The input is not overwritten unless you explicitly use the same path.

## Configuration

All parameters live in `config/scanner.yaml`:

- `board`: physical ChArUco dimensions and detector thresholds.
- `capture.record_sample_hz`: reconstruction sample rate for recording/replay.
- `object.min_height_m`: reject the board/background.
- `reconstruction.voxel_size_m`: TSDF detail versus memory.
- `visualizer.poses`: desired render angles.
- `visualizer.contact_sheet_columns`: layout of the combined multi-view image.
- `decimation.target_triangles`: default low-poly triangle count.

## Limitations

- The board must remain visible in frames used for reconstruction.
- Keep the object and board stationary during a recording.
- The underside is hidden by the board; scanning it requires a second scan and a future scan-merging step.
- Reflective, transparent, very dark or thin objects can have missing depth.
- This produces approximate geometry, not metrology-grade CAD.
