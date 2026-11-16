from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

from scanner.diagnostics import (
    evaluate_scan_quality,
    mesh_measurements,
    remove_depth_outliers,
    save_contact_sheet,
    save_segmentation_preview,
)


def _object_depth_mask(
    depth: np.ndarray,
    transform_camera_board: np.ndarray,
    metadata: dict,
    config: dict,
) -> np.ndarray:
    """Keep the single target object above the board, inside board bounds and valid range."""
    scale = float(metadata["depth_scale_m_per_unit"])
    intr = metadata["intrinsics"]
    height, width = depth.shape
    depth_m = depth.astype(np.float32) * scale
    camera_config = config["camera"]
    minimum_depth = float(camera_config.get("min_depth_m", 0.0))
    maximum_depth = float(camera_config["max_depth_m"])
    valid = (depth_m >= minimum_depth) & (depth_m <= maximum_depth) & (depth > 0)

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


def _cleanup_mesh(mesh: o3d.geometry.TriangleMesh, settings: dict) -> dict:
    """Clean degenerate/duplicate faces and optionally remove tiny components."""
    before_vertices = len(mesh.vertices)
    before_triangles = len(mesh.triangles)

    if settings.get("remove_degenerate", True):
        mesh.remove_degenerate_triangles()
    if settings.get("remove_duplicate", True):
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    components_before = 0
    components_removed = 0
    if len(mesh.triangles) and settings.get("remove_small_components", True):
        labels, counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(counts, dtype=np.int64)
        components_before = len(counts)
        if len(counts):
            minimum = int(settings.get("min_component_triangles", 30))
            remove = counts[np.asarray(labels)] < minimum
            if settings.get("keep_largest_component_only", False):
                remove |= np.asarray(labels) != int(np.argmax(counts))
            components_removed = len(counts) - len(
                np.unique(np.asarray(labels)[~remove])
            )
            mesh.remove_triangles_by_mask(remove)
            mesh.remove_unreferenced_vertices()

    smooth_iterations = max(0, int(settings.get("smooth_iterations", 0)))
    if smooth_iterations and len(mesh.triangles):
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)

    if settings.get("remove_non_manifold_edges", False) and len(mesh.triangles):
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()

    if len(mesh.triangles):
        mesh.compute_vertex_normals()

    return {
        "vertices_before": before_vertices,
        "triangles_before": before_triangles,
        "vertices_after": len(mesh.vertices),
        "triangles_after": len(mesh.triangles),
        "components_before": components_before,
        "components_removed": components_removed,
        "smooth_iterations": smooth_iterations,
    }


def reconstruct(dataset: str, output: str, config: dict, show: bool = True) -> None:
    """Fuse saved RGB-D frames and write segmentation, quality and mesh reports."""
    source = Path(dataset)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))

    intr = metadata["intrinsics"]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        int(intr["width"]),
        int(intr["height"]),
        float(intr["fx"]),
        float(intr["fy"]),
        float(intr["cx"]),
        float(intr["cy"]),
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

    diagnostics_dir = destination / "diagnostics"
    preview_dir = diagnostics_dir / "segmentation"
    preview_config = config.get("segmentation_preview", {})
    previews = []
    preview_limit = max(0, int(preview_config.get("max_frames", 12)))
    outlier_stats = []
    segmentation_stats = []
    integrated = 0
    skipped_empty = 0

    for frame in metadata.get("frames", []):
        frame_id = frame["id"]
        color_bgr = cv2.imread(str(source / "color" / f"{frame_id}.png"))
        depth = cv2.imread(
            str(source / "depth" / f"{frame_id}.png"), cv2.IMREAD_UNCHANGED
        )
        pose_path = source / "poses" / f"{frame_id}.npy"
        if color_bgr is None or depth is None or not pose_path.exists():
            print(f"Skipping incomplete frame {frame_id}")
            continue
        transform = np.load(pose_path)

        raw_masked_depth = _object_depth_mask(depth, transform, metadata, config)
        masked_depth, filter_stats = remove_depth_outliers(
            raw_masked_depth,
            float(metadata["depth_scale_m_per_unit"]),
            intr,
            config,
        )
        raw_pixels = int(np.count_nonzero(raw_masked_depth))
        kept_pixels = int(np.count_nonzero(masked_depth))
        segmentation_stats.append({
            "frame_id": frame_id,
            "object_pixels_before_outlier_removal": raw_pixels,
            "object_pixels_after_outlier_removal": kept_pixels,
            "retained_percent": (
                100.0 * kept_pixels / raw_pixels if raw_pixels else 0.0
            ),
        })
        filter_stats["frame_id"] = frame_id
        outlier_stats.append(filter_stats)

        if preview_config.get("enabled", True) and len(previews) < preview_limit:
            overlay = save_segmentation_preview(
                preview_dir,
                frame_id,
                color_bgr,
                raw_masked_depth,
                masked_depth,
                float(metadata["depth_scale_m_per_unit"]),
                float(config["camera"]["max_depth_m"]),
            )
            previews.append(overlay)

        if kept_pixels < int(config.get("scan_quality", {}).get("min_object_pixels", 100)):
            skipped_empty += 1
            print(
                f"Skipped {frame_id}: only {kept_pixels} object pixels remain "
                "after segmentation/outlier rejection."
            )
            continue

        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_rgb),
            o3d.geometry.Image(masked_depth),
            depth_scale=1.0 / float(metadata["depth_scale_m_per_unit"]),
            depth_trunc=float(config["camera"]["max_depth_m"]),
            convert_rgb_to_intensity=False,
        )

        # ChArUco provides T_camera_board (board/world -> camera), as TSDF expects.
        volume.integrate(rgbd, intrinsic, transform)
        integrated += 1
        print(
            f"Integrated {frame_id} ({integrated}/{len(metadata.get('frames', []))}); "
            f"object pixels {kept_pixels}"
        )

    if previews:
        save_contact_sheet(
            previews,
            diagnostics_dir / "segmentation_preview.png",
            int(preview_config.get("columns", 3)),
        )

    (diagnostics_dir / "segmentation_stats.json").write_text(
        json.dumps(segmentation_stats, indent=2), encoding="utf-8"
    )
    (diagnostics_dir / "outlier_stats.json").write_text(
        json.dumps(outlier_stats, indent=2), encoding="utf-8"
    )

    scan_quality = evaluate_scan_quality(metadata, config, source)
    scan_quality["integrated_views"] = integrated
    scan_quality["skipped_views_with_insufficient_object_pixels"] = skipped_empty
    (diagnostics_dir / "scan_quality.json").write_text(
        json.dumps(scan_quality, indent=2), encoding="utf-8"
    )
    for warning in scan_quality["warnings"]:
        print(f"SCAN QUALITY WARNING: {warning}")

    if integrated == 0:
        raise RuntimeError(
            "No frames were integrated. Check the minimum depth, object height, "
            "outlier settings, ChArUco tracking and camera viewpoints."
        )

    mesh = volume.extract_triangle_mesh()
    if len(mesh.triangles) == 0:
        raise RuntimeError(
            "No mesh surface was reconstructed. Check object segmentation, near/far depth "
            "limits, camera viewpoints and whether the object is above the board."
        )

    cleanup_stats = _cleanup_mesh(mesh, config.get("mesh_cleanup", {}))
    if len(mesh.triangles) == 0:
        raise RuntimeError(
            "Mesh cleanup removed all triangles. Lower mesh_cleanup.min_component_triangles."
        )

    before_decimation = {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
    }
    target = int(reconstruction_config["target_triangles"])
    if reconstruction_config.get("simplify", True) and len(mesh.triangles) > target:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()

    if not len(mesh.triangles):
        raise RuntimeError("Mesh decimation produced an empty mesh.")

    o3d.io.write_triangle_mesh(str(destination / "mesh.ply"), mesh, write_vertex_colors=True)
    o3d.io.write_triangle_mesh(str(destination / "mesh.obj"), mesh, write_vertex_colors=True)
    o3d.io.write_triangle_mesh(str(destination / "mesh.stl"), mesh, write_vertex_colors=False)

    # Preserve the exact settings used so the same .bag/dataset can be reconstructed again.
    (destination / "reconstruction_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    total_raw = sum(item["input_pixels"] for item in outlier_stats)
    total_kept = sum(item["output_pixels"] for item in outlier_stats)
    total_median_rejected = sum(item["median_rejected"] for item in outlier_stats)
    total_stat_rejected = sum(item["statistical_rejected"] for item in outlier_stats)

    report = {
        "source_dataset": str(source.resolve()),
        "integrated_views": integrated,
        "input_views": len(metadata.get("frames", [])),
        "skipped_views_with_insufficient_object_pixels": skipped_empty,
        "scan_quality": scan_quality,
        "segmentation": {
            "assumption": "single_target_on_board",
            "method": "board_height_and_bounds",
            "object_pixels_before_outlier_removal": total_raw,
            "object_pixels_after_outlier_removal": total_kept,
            "retained_percent": 100.0 * total_kept / total_raw if total_raw else 0.0,
        },
        "outlier_removal": {
            "enabled": bool(config.get("outlier_removal", {}).get("enabled", False)),
            "median_rejected_pixels": total_median_rejected,
            "statistical_rejected_pixels": total_stat_rejected,
            "rejected_percent": (
                100.0 * (total_raw - total_kept) / total_raw if total_raw else 0.0
            ),
            "per_frame": outlier_stats,
        },
        "mesh_cleanup": cleanup_stats,
        "mesh_before_decimation": before_decimation,
        "mesh": mesh_measurements(mesh),
        "settings": config,
        "notes": [
            "Depth samples closer than camera.min_depth_m are rejected.",
            "Outlier filtering cannot recover invalid measurements inside the sensor's unreliable near range.",
            "Dimensions are estimates from the reconstructed mesh, not metrology-grade measurements.",
            "The board-contact underside is not observed in a single scan.",
        ],
    }
    (destination / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(
        f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles"
    )
    print(
        "Approx dimensions (AABB, mm): "
        + " x ".join(f"{value:.1f}" for value in report["mesh"]["dimensions_aabb_mm"])
    )
    print(f"Reports: {destination / 'report.json'}")
    print(f"Segmentation diagnostics: {diagnostics_dir.resolve()}")
    print(f"Exported PLY / OBJ / STL to {destination.resolve()}")

    if show:
        o3d.visualization.draw_geometries([mesh], window_name="Low-poly object scan")
