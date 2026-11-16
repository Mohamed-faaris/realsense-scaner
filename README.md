# RealSense Scaner

A small D415 object-scanning tool. Move the camera around an object placed on a ChArUco board. OpenCV detects the board automatically and estimates camera pose. Open3D fuses saved RGB-D views into a mesh.

The project stays Python-first: CLI, YAML configuration, OpenCV preview, and Open3D visualization. No web UI or database.

## Stack

- Python 3.11 + `uv`
- RealSense SDK / `pyrealsense2`
- OpenCV contrib (automatic ChArUco detection)
- Open3D (TSDF, mesh simplification, rendering)
- YAML configuration

## Board

Configured for the existing board: 5 × 7 squares, 37 mm squares, 23.5 mm markers, 185 × 259 mm overall, `DICT_5X5_50`, 17 ArUco markers and 24 ChArUco inner corners.

Generate a fresh image from the configuration:

```bash
uv run rs-scan board --output charuco-board.png
```

Print at actual size and measure the squares before capture.

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

## Capture RGB-D

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

This stores the following in one session directory:

- `recording.mp4`: continuous RGB video for review.
- `color/*.png`: lossless RGB images for each accepted reconstruction sample.
- `depth/*.png`: aligned 16-bit depth images.
- `poses/*.npy`: ChArUco camera pose for each sample.
- `metadata.json`: camera intrinsics, depth scale, frame timestamps, video frame indices and board settings.

Valid reconstruction samples are saved automatically at `capture.record_sample_hz` (default 5 Hz). SPACE forces an additional sample. Frames where ChArUco tracking fails remain in the MP4 for review but are not used for reconstruction.

**Important:** an ordinary RGB MP4 alone cannot produce metric RGB-D geometry. Keep the complete session directory. The synchronized depth frames, camera poses, intrinsics and depth scale are what make later reconstruction possible without connecting the D415 again.

## Reconstruct later (no camera required)

```bash
uv run rs-scan reconstruct \
  --dataset datasets/scan01 \
  --output outputs/scan01
```

This reads only the saved session data. It does not open the camera.

For headless reconstruction:

```bash
uv run rs-scan reconstruct --dataset datasets/scan01 \
  --output outputs/scan01 --no-view
```

Outputs: `mesh.ply`, `mesh.obj`, `mesh.stl`.

## Render multiple views into one image

Edit `visualizer.poses` in `config/scanner.yaml`. Each pose has a name, azimuth, elevation and distance multiplier.

```bash
uv run rs-scan visualize \
  --mesh outputs/scan01/mesh.ply \
  --output outputs/scan01/views
```

Creates one PNG per configured angle and a combined `all_views.png` contact sheet containing every configured pose.

Default views: front, right, rear, left, top and isometric. Rendering uses Open3D OffscreenRenderer; on Linux, a working OpenGL/EGL graphics runtime is required.

## Decimate a mesh

```bash
uv run rs-scan decimate \
  --input outputs/scan01/mesh.ply \
  --output outputs/scan01/mesh_decimated.ply \
  --target-triangles 1500
```

The default target comes from `decimation.target_triangles` in YAML. Quadric-error decimation writes the simplified mesh and a JSON report with before/after vertex and triangle counts. The input mesh is never overwritten unless the same path is explicitly supplied.

## Useful configuration

All scanner parameters live in `config/scanner.yaml`:

- `board`: physical ChArUco dimensions and tracking thresholds.
- `capture.record_sample_hz`: reconstruction sample frequency during video recording.
- `object.min_height_m`: board/background rejection threshold.
- `reconstruction.voxel_size_m`: TSDF detail versus memory.
- `visualizer.poses`: output camera angles.
- `visualizer.contact_sheet_columns`: number of columns in the combined view image.
- `decimation.target_triangles`: default low-poly target.

## Limitations

- The board must stay visible in frames used for reconstruction.
- The object and board must remain stationary during capture.
- The underside is hidden by the board; a second scan and scan-merging workflow is needed to capture it.
- Reflective, transparent, very dark and thin objects may have missing depth.
- The output is approximate geometry, not metrology-grade CAD.
