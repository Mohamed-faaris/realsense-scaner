from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def _object_depth_mask(
    depth: np.ndarray,
    transform_camera_board: np.ndarray,
    metadata: dict,
    config: dict,
) -> np.ndarray:
    """Keep depth pixels above the board, on the same side as the camera."""
    scale = float(metadata["depth_scale_m_per_unit"])
    intr = metadata["intrinsics"]
    height, width = depth.shape
    depth_m = depth.astype(np.float32) * scale
    valid = (depth_m > 0) & (depth_m <= float(config["camera"]["max_depth_m"]))

    v, u = np.indices((height, width), dtype=np.float32)
    x = (u - intr["cx"]) * depth_m / intr["fx"]
    y = (v - intr["cy"]) * depth_m / intr["fy"]

    points_camera = np.stack((x, y, depth_m, np.ones_like(depth_m)), axis=-1)
    transform_board_camera = np.linalg.inv(transform_camera_board)
    points_board = points_camera @ transform_board_camera.T

    bx, by, bz = points_board[..., 0], points_board[..., 1], points_board[..., 2]
    camera_origin_board = transform_board_camera[:3, 3]
    side = 1.0 if camera_origin_board[2] >= 0 else -1.0
    height_above_board = bz * side

    board_width = float(metadata["board_width_m"])
    board_height = float(metadata["board_height_m"])
    object_config = config["object"]
    keep = valid
    keep &= height_above_board >= float(object_config["min_height_m"])
    keep &= height_above_board <= float(object_config["max_height_m"])
    keep &= (bx >= 0) & (bx <= board_width)
    keep &= (by >= 0) & (by <= board_height)

    roi = object_config.get("roi_xy_m")
    if roi:
        xmin, ymin, xmax, ymax = map(float, roi)
        keep &= (bx >= xmin) & (bx <= xmax) & (by >= ymin) & (by <= ymax)

    masked = np.zeros_like(depth)
    masked[keep] = depth[keep]
    return masked


def reconstruct(dataset: str, output: str, config: dict, show: bool = True) -> None:
    """Fuse saved RGB-D frames using their ChArUco board poses."""
    source = Path(dataset)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))

    intr = metadata["intrinsics"]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        int(intr["width"]), int(intr["height"]),
        float(intr["fx"]), float(intr["fy"]),
        float(intr["cx"]), float(intr["cy"]),
    )
    reconstruction_config = config["reconstruction"]
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(reconstruction_config["voxel_size_m"]),
        sdf_trunc=float(reconstruction_config["sdf_trunc_m"]),
        color_type=(
            o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            if reconstruction_config.get("integrate_color", True)
            else o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        ),
    )

    integrated = 0
    for frame in metadata["frames"]:
        frame_id = frame["id"]
        color_bgr = cv2.imread(str(source / "color" / f"{frame_id}.png"))
        depth = cv2.imread(
            str(source / "depth" / f"{frame_id}.png"), cv2.IMREAD_UNCHANGED
        )
        transform = np.load(source / "poses" / f"{frame_id}.npy")
        if color_bgr is None or depth is None:
            print(f"Skipping incomplete frame {frame_id}")
            continue

        masked_depth = _object_depth_mask(depth, transform, metadata, config)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_rgb),
            o3d.geometry.Image(masked_depth),
            depth_scale=1.0 / float(metadata["depth_scale_m_per_unit"]),
            depth_trunc=float(config["camera"]["max_depth_m"]),
            convert_rgb_to_intensity=False,
        )

        # The board is the world frame; this is T_camera_board (world -> camera).
        volume.integrate(rgbd, intrinsic, transform)
        integrated += 1
        print(f"Integrated {frame_id} ({integrated}/{len(metadata['frames'])})")

    if integrated == 0:
        raise RuntimeError("No frames were integrated. Check the dataset and board tracking.")

    mesh = volume.extract_triangle_mesh()
    if len(mesh.triangles) == 0:
        raise RuntimeError(
            "No mesh surface was reconstructed. Check object segmentation, depth range, "
            "camera viewpoints and whether the object is actually above the board."
        )

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    target = int(reconstruction_config["target_triangles"])
    if reconstruction_config.get("simplify", True) and len(mesh.triangles) > target:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(str(destination / "mesh.ply"), mesh, write_vertex_colors=True)
    o3d.io.write_triangle_mesh(str(destination / "mesh.obj"), mesh, write_vertex_colors=True)
    o3d.io.write_triangle_mesh(str(destination / "mesh.stl"), mesh, write_vertex_colors=False)

    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    print(f"Exported PLY / OBJ / STL to {destination.resolve()}")
    if show:
        o3d.visualization.draw_geometries([mesh], window_name="Low-poly object scan")
