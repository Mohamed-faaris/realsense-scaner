# RealSense Scaner

A small D415 object-scanning tool. Move the camera around one stationary object placed on a ChArUco board. OpenCV detects the board automatically and estimates camera pose. Open3D fuses RGB-D views into an approximate low-poly mesh.

Python-first: CLI, YAML configuration, OpenCV preview and Open3D visualization. No frontend framework or database.

## Stack

- Python 3.11 + `uv`
- RealSense SDK / `pyrealsense2`
- OpenCV contrib for automatic ChArUco detection
- Open3D for TSDF, point-cloud filtering, mesh cleanup, decimation and rendering
- YAML configuration

## Board

Configured for the physical board: 5 × 7 squares, 37 mm squares, 23.5 mm markers, 185 × 259 mm overall, `DICT_5X5_50`, 17 ArUco markers and 24 ChArUco inner corners.

Generate the configured board image:

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

## Capture RGB-D

Manual viewpoint capture:

```bash
uv run rs-scan capture --output datasets/scan01
```

- Move the camera slowly around the stationary object.
- Keep at least 8 ChArUco corners visible.
- Press SPACE to save a valid view.
- Press Q or ESC to finish.

Continuous recording:

```bash
uv run rs-scan capture --output datasets/scan01 --record-video
```

This records a complete session:

- `recording.bag`: raw synchronized RealSense color + depth streams; the reusable 3D source.
- `recording.mp4`: continuous RGB review video.
- `color/*.png`: RGB samples.
- `depth/*.png`: aligned 16-bit depth samples.
- `poses/*.npy`: ChArUco camera pose for each accepted sample.
- `metadata.json`: intrinsics, depth scale, timestamps, board dimensions and sample mapping.

Valid RGB-D views are sampled at `capture.record_sample_hz` (default 5 Hz). SPACE forces an additional sample. MP4 continues recording if tracking is lost, but those frames are not accepted for reconstruction.

**MP4 is not a metric 3D source.** It has no depth. Keep the `.bag` file to reprocess the original depth and color later.

## Reprocess a recording (camera not required)

Re-run ChArUco detection with the bag file:

```bash
uv run rs-scan replay-bag \
  --bag datasets/scan01/recording.bag \
  --output datasets/scan01_reprocessed
```

Then reconstruct normally:

```bash
uv run rs-scan reconstruct \
  --dataset datasets/scan01_reprocessed \
  --output outputs/scan01
```

The RealSense bag stores synchronized streams and can be replayed using librealsense. The physical D415 is not required for replay or reconstruction.

## Reconstruction, previews and outlier filtering

```bash
uv run rs-scan reconstruct \
  --dataset datasets/scan01 \
  --output outputs/scan01
```

The segmentation assumes **one target object on the board**:

1. Reject depth outside `camera.min_depth_m` and `camera.max_depth_m`.
2. Transform remaining points into ChArUco board coordinates.
3. Keep points above `object.min_height_m` and inside board bounds.
4. Remove isolated depth spikes using the configurable median filter.
5. Apply 3D statistical outlier removal (enabled by default) to reject spatially isolated points.
6. Integrate accepted depth frames using TSDF.

To disable outlier removal for one reconstruction:

```bash
uv run rs-scan reconstruct --dataset datasets/scan01 \
  --output outputs/no_filter --no-outlier-removal
```

The default median filter uses a 3×3 neighborhood and rejects local depth deviations over 20 mm. **3D statistical outlier removal is also enabled by default** with 20 neighbors and a 2.0 standard-deviation ratio. It converts the remaining object depth pixels into XYZ points, removes spatially isolated points, and masks those pixels before TSDF fusion. Statistical filtering can remove legitimate sparse details, so compare the segmentation previews and `diagnostics/outlier_stats.json` before tightening its settings.

Tune it in YAML:

```yaml
outlier_removal:
  enabled: true
  statistical:
    enabled: true
    nb_neighbors: 20
    std_ratio: 2.0
    min_points: 100
```

Increase `std_ratio` to preserve more points; decrease it to reject more outliers. Increase `nb_neighbors` for a broader neighborhood test. These values depend on point density and camera distance.

**Near-range warning:** software cannot correct depth measurements made inside the camera's unreliable minimum range. Points closer than `camera.min_depth_m` are rejected; increase camera-to-object distance or tune the threshold after testing the D415 at your actual working distance.

### Segmentation preview

Every reconstruction writes:

- `diagnostics/segmentation/*_mask.png`: retained object pixels (white), rejected outlier pixels (gray).
- `diagnostics/segmentation/*_overlay.png`: retained object pixels highlighted green and rejected depth outliers red.
- `diagnostics/segmentation/*_depth.png`: segmented depth visualization.
- `diagnostics/segmentation_preview.png`: combined preview of several input views.
- `diagnostics/segmentation_stats.json`: per-frame retained pixel counts.
- `diagnostics/outlier_stats.json`: rejected depth pixel counts by filter.

### Scan quality

`diagnostics/scan_quality.json` reports valid frames, approximate azimuth-sector coverage, and mean/max ChArUco reprojection error. Warnings identify too few views, limited angular coverage, or poor board tracking. Coverage is a heuristic based on camera positions in board coordinates, not a guarantee of complete surface coverage.

### Mesh cleanup and measurement report

Reconstruction removes degenerate/duplicate triangles and can remove tiny disconnected mesh components. Optional Taubin smoothing is configurable but disabled by default.

`report.json` contains:

- Estimated AABB and oriented bounding-box dimensions in mm.
- Vertex and triangle counts.
- Surface area, watertight status and volume when watertight.
- Segmentation retention and outlier statistics.
- Scan quality and mesh cleanup results.
- All settings used for the reconstruction.

The exact configuration is also saved as `reconstruction_config.yaml`, so you can rerun reconstruction from the same dataset or bag after changing parameters.

## Render configured views

Edit `visualizer.poses` in YAML. Each pose specifies azimuth, elevation and distance.

```bash
uv run rs-scan visualize \
  --mesh outputs/scan01/mesh.ply \
  --output outputs/scan01/views
```

Creates per-angle PNG files and a combined `all_views.png` contact sheet. Default views: front, right, rear, left, top and isometric. Open3D offscreen rendering needs a working OpenGL/EGL runtime on Linux.

## Decimate a mesh

```bash
uv run rs-scan decimate \
  --input outputs/scan01/mesh.ply \
  --output outputs/scan01/mesh_decimated.ply \
  --target-triangles 1500
```

Without `--target-triangles`, the target comes from `decimation.target_triangles`. Quadric-error decimation writes a simplified mesh and a JSON report with before/after counts. The input is not overwritten unless the same path is explicitly selected.

## Configuration reference

All settings live in `config/scanner.yaml`:

- `camera.min_depth_m`: reject too-near measurements.
- `outlier_removal`: median and 3D statistical filtering (both enabled by default).
- `segmentation_preview`: enable previews and set number of saved frames.
- `scan_quality`: minimum frame count, view coverage and tracking thresholds.
- `mesh_cleanup`: remove tiny components and optionally smooth.
- `visualizer.poses`: saved image viewpoints.
- `decimation.target_triangles`: default low-poly budget.

## Limitations

- The board must remain visible in frames used for ChArUco pose estimation.
- Only one target object is assumed; it must fit within the board bounds.
- Keep the object and board stationary during a scan.
- The board-contact underside is not observed; a second scan and scan-merging workflow is needed.
- Reflective, transparent, very dark and thin objects can have missing depth.
- This is approximate reconstruction, not metrology-grade CAD.
