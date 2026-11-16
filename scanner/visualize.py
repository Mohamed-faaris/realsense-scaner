from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from open3d.visualization import rendering


def _camera_eye(center: np.ndarray, radius: float, pose: dict) -> tuple[np.ndarray, np.ndarray]:
    azimuth = math.radians(float(pose["azimuth_deg"]))
    elevation = math.radians(float(pose["elevation_deg"]))
    direction = np.array([
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ], dtype=np.float64)
    eye = center + direction * radius * float(pose.get("distance", 1.0))
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(direction, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    return eye, up


def visualize_mesh(mesh_path: str, output_dir: str, config: dict) -> Path:
    """Render configured camera angles and make a single contact-sheet image."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise ValueError(f"No triangle mesh found at {mesh_path}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    visual = config["visualizer"]
    poses = visual["poses"]
    if not poses:
        raise ValueError("visualizer.poses must contain at least one camera pose")

    bounds = mesh.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_extent()
    diagonal = max(float(np.linalg.norm(extent)), 1e-3)
    fov = float(visual.get("fov_deg", 45.0))
    base_distance = (diagonal * 0.5 / math.sin(math.radians(fov * 0.5))) * 1.15

    width = int(visual.get("width", 800))
    height = int(visual.get("height", 600))
    renderer = rendering.OffscreenRenderer(width, height)
    background = np.asarray(visual.get("background_rgba", [0.96, 0.97, 0.98, 1.0]),
                            dtype=np.float32)
    renderer.scene.set_background(background)

    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.92, 0.94, 0.97, 1.0]
    renderer.scene.add_geometry("scanned_mesh", mesh, material)

    if visual.get("show_axes", True):
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=max(float(np.max(extent)) * 0.22, 0.02),
            origin=center,
        )
        axis_material = rendering.MaterialRecord()
        axis_material.shader = "defaultLit"
        renderer.scene.add_geometry("world_axes", axis, axis_material)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[str, np.ndarray]] = []

    try:
        for pose in poses:
            eye, up = _camera_eye(center, base_distance, pose)
            renderer.setup_camera(
                fov,
                center.astype(np.float32),
                eye.astype(np.float32),
                up.astype(np.float32),
                max(base_distance * 0.001, 0.001),
                base_distance * 10.0,
            )
            rgb = np.asarray(renderer.render_to_image())
            if rgb.ndim != 3 or rgb.shape[2] < 3:
                raise RuntimeError(f"Unexpected render output for pose {pose['name']}")
            bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
            title = str(pose["name"]).replace("_", " ").title()
            cv2.rectangle(bgr, (0, 0), (width, 42), (245, 245, 245), -1)
            cv2.putText(bgr, title, (16, 29), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (35, 40, 48), 2, cv2.LINE_AA)
            safe_name = "".join(c.lower() if c.isalnum() else "_" for c in str(pose["name"]))
            image_path = out / f"{safe_name}.png"
            if not cv2.imwrite(str(image_path), bgr):
                raise OSError(f"Could not save render: {image_path}")
            rendered.append((title, bgr))
            print(f"Rendered {title}: {image_path}")
    finally:
        del renderer

    columns = max(1, int(visual.get("contact_sheet_columns", 2)))
    rows = math.ceil(len(rendered) / columns)
    tile_h, tile_w = height, width
    sheet = np.full(
        (rows * tile_h, columns * tile_w, 3),
        248,
        dtype=np.uint8,
    )
    for index, (_, image) in enumerate(rendered):
        row, column = divmod(index, columns)
        sheet[row * tile_h:(row + 1) * tile_h,
              column * tile_w:(column + 1) * tile_w] = image

    sheet_path = out / "all_views.png"
    if not cv2.imwrite(str(sheet_path), sheet):
        raise OSError(f"Could not save contact sheet: {sheet_path}")
    print(f"Combined all {len(rendered)} poses into {sheet_path.resolve()}")
    return sheet_path
