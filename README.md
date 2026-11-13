# RealSense Scaner

A deliberately small D415 object-scanning project. The camera moves around a stationary object placed on a ChArUco board. OpenCV detects the board in each saved frame and estimates the camera pose. Open3D fuses the RGB-D views into a low-poly mesh.

No SLAM, frontend framework, database, web server, or custom reconstruction engine.

## Stack

- Python 3.11
- `uv` for the environment and dependencies
- `pyrealsense2` for D415 capture
- OpenCV ArUco for automatic ChArUco detection and camera pose
- Open3D for TSDF fusion and mesh simplification
- YAML for scanner configuration

## Board configuration

This project is configured for the existing board:

| Parameter | Value |
|---|---:|
| Squares | 5 × 7 |
| Square size | 37 mm |
| Marker size | 23.5 mm |
| Physical size | 185 × 259 mm |
| Dictionary | `DICT_5X5_50` |
| ArUco markers | 17 |
| ChArUco inner corners | 24 |

The board is generated from `config/scanner.yaml`; the detector uses the same configuration. Print at actual size and measure the squares.

## Setup

Install the RealSense SDK / udev rules for your Ubuntu release first:
https://github.com/realsenseai/librealsense

Then:

```bash
git clone https://github.com/Mohamed-faaris/realsense-scaner.git
cd realsense-scaner

uv python install 3.11
uv sync --extra hardware
```

If the `pyrealsense2` wheel is unavailable for your Python/Ubuntu combination, install the binding using the librealsense instructions and run `uv sync` without the hardware extra.

## Workflow

Generate the exact board image:

```bash
uv run rs-scan board --output charuco-board.png
```

Capture views:

```bash
uv run rs-scan capture --output datasets/scan01
```

The preview automatically detects the ChArUco board and displays detected corners, coordinate axes, and reprojection error.

- Move the camera slowly between viewpoints.
- Keep at least 8 ChArUco corners visible.
- Press SPACE to save the current valid view.
- Press Q or ESC to finish.
- Capture the top and multiple side angles.
- The object must remain still during a scan.

Reconstruct:

```bash
uv run rs-scan reconstruct --dataset datasets/scan01 --output outputs/scan01
```

Run headless (without opening the Open3D viewer):

```bash
uv run rs-scan reconstruct --dataset datasets/scan01 --output outputs/scan01 --no-view
```

Outputs are `mesh.ply`, `mesh.obj`, and `mesh.stl`.

## Tuning

Edit only `config/scanner.yaml`:

- `board`: exact printed board dimensions and tracking thresholds.
- `object.min_height_m`: suppress board depth noise.
- `object.roi_xy_m`: optional object crop in board coordinates.
- `reconstruction.voxel_size_m`: geometry resolution.
- `reconstruction.target_triangles`: low-poly mesh budget.

Start with 3 mm voxels and 3,000 triangles.

## Coordinate handling

Each accepted frame stores `T_camera_board` from OpenCV. During reconstruction, depth pixels are transformed into board coordinates to segment points above the board. Open3D TSDF integration receives `T_camera_board` as its world-to-camera extrinsic.

The object-side sign is inferred from the camera position relative to the board, avoiding assumptions about the board's normal direction.

## Scope / limitations

- The board must stay visible; frames without a valid pose are not saved.
- The board and object must remain rigid/stationary.
- The object must fit within the board area for the default segmentation.
- The underside is hidden by the board. Flip the object and perform another scan if the underside is required; merging flipped scans is not automated in this first version.
- Reflective, transparent, very dark, or thin objects can have missing depth.
- This produces an approximate scan, not metrology-grade CAD.
- The UI is intentionally just an OpenCV capture preview and Open3D viewer. Build a richer UI against these small Python modules later.
