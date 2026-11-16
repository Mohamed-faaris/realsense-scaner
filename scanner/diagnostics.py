from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def remove_depth_outliers(
    depth: np.ndarray,
    depth_scale_m_per_unit: float,
    intrinsics: dict,
    config: dict,
) -> tuple[np.ndarray, dict]:
    """Reject local depth spikes and optionally statistical 3D outliers.

    Input depth must already be masked to the target object. Measurements outside
    the configured camera depth range should be removed before calling this function.
    """
    settings = config.get("outlier_removal", {})
    result = depth.copy()
    input_count = int(np.count_nonzero(result))
    stats = {
        "input_pixels": input_count,
        "median_rejected": 0,
        "statistical_rejected": 0,
        "output_pixels": input_count,
        "enabled": bool(settings.get("enabled", False)),
    }
    if not stats["enabled"] or input_count == 0:
        return result, stats

    scale = float(depth_scale_m_per_unit)
    if scale <= 0:
        raise ValueError("Depth scale must be positive")

    median_cfg = settings.get("median", {})
    if median_cfg.get("enabled", True):
        kernel = int(median_cfg.get("kernel_size", 3))
        if kernel < 3 or kernel % 2 == 0:
            raise ValueError("outlier_removal.median.kernel_size must be an odd value >= 3")
        local_median = cv2.medianBlur(result, kernel)
        valid_median = local_median > 0
        max_deviation_units = (
            float(median_cfg.get("max_deviation_mm", 20.0)) / 1000.0 / scale
        )
        outliers = (
            (result > 0)
            & valid_median
            & (np.abs(result.astype(np.int32) - local_median.astype(np.int32))
               > max_deviation_units)
        )
        stats["median_rejected"] = int(np.count_nonzero(outliers))
        result[outliers] = 0

    statistical_cfg = settings.get("statistical", {})
    remaining = int(np.count_nonzero(result))
    if statistical_cfg.get("enabled", False) and remaining >= int(
        statistical_cfg.get("min_points", 100)
    ):
        intr = intrinsics
        v, u = np.nonzero(result)
        z = result[v, u].astype(np.float64) * scale
        x = (u.astype(np.float64) - float(intr["cx"])) * z / float(intr["fx"])
        y = (v.astype(np.float64) - float(intr["cy"])) * z / float(intr["fy"])
        points = np.column_stack((x, y, z))
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        nb_neighbors = max(3, int(statistical_cfg.get("nb_neighbors", 20)))
        std_ratio = float(statistical_cfg.get("std_ratio", 2.0))
        if len(points) > nb_neighbors:
            _, inlier_indices = cloud.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio,
            )
            keep = np.zeros(len(points), dtype=bool)
            keep[np.asarray(inlier_indices, dtype=np.int64)] = True
            rejected_pixels = ~keep
            stats["statistical_rejected"] = int(np.count_nonzero(rejected_pixels))
            result[v[rejected_pixels], u[rejected_pixels]] = 0

    stats["output_pixels"] = int(np.count_nonzero(result))
    return result, stats


def save_segmentation_preview(
    output_dir: Path,
    frame_id: str,
    color_bgr: np.ndarray,
    raw_masked_depth: np.ndarray,
    cleaned_depth: np.ndarray,
    depth_scale_m_per_unit: float,
    max_depth_m: float,
) -> np.ndarray:
    """Write per-frame object mask, depth view and RGB overlay; return overlay."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = cleaned_depth > 0
    rejected = (raw_masked_depth > 0) & ~selected

    mask_image = np.zeros(cleaned_depth.shape, dtype=np.uint8)
    mask_image[selected] = 255
    mask_image[rejected] = 100
    cv2.imwrite(str(output_dir / f"{frame_id}_mask.png"), mask_image)

    depth_m = cleaned_depth.astype(np.float32) * float(depth_scale_m_per_unit)
    depth_gray = np.clip(depth_m / max(float(max_depth_m), 1e-6) * 255.0, 0, 255)
    cv2.imwrite(
        str(output_dir / f"{frame_id}_depth.png"),
        depth_gray.astype(np.uint8),
    )

    overlay = color_bgr.copy()
    tint = overlay.copy()
    tint[selected] = (50, 210, 70)  # retained target pixels
    tint[rejected] = (30, 30, 240)  # rejected depth outliers
    overlay = cv2.addWeighted(overlay, 0.55, tint, 0.45, 0)
    cv2.putText(
        overlay,
        f"Object pixels: {np.count_nonzero(selected)} | rejected: {np.count_nonzero(rejected)}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_dir / f"{frame_id}_overlay.png"), overlay)
    return overlay


def save_contact_sheet(images: list[np.ndarray], output_path: Path, columns: int) -> None:
    if not images:
        return
    columns = max(1, int(columns))
    tile_h, tile_w = images[0].shape[:2]
    rows = math.ceil(len(images) / columns)
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 245, dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        sheet[row * tile_h:(row + 1) * tile_h, column * tile_w:(column + 1) * tile_w] = image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise OSError(f"Could not write segmentation contact sheet: {output_path}")


def evaluate_scan_quality(metadata: dict, config: dict, dataset_dir: Path) -> dict:
    """Estimate scan viewpoint coverage and ChArUco tracking quality."""
    settings = config.get("scan_quality", {})
    poses = []
    reprojection_errors = []
    tracking_corners = []
    for frame in metadata.get("frames", []):
        frame_id = frame["id"]
        pose_path = dataset_dir / "poses" / f"{frame_id}.npy"
        if not pose_path.is_file():
            continue
        transform = np.load(pose_path)
        camera_origin_board = np.linalg.inv(transform)[:3, 3]
        x, y, z = map(float, camera_origin_board)
        radius_xy = math.hypot(x, y)
        azimuth = math.degrees(math.atan2(y, x))
        elevation = math.degrees(math.atan2(z, max(radius_xy, 1e-9)))
        poses.append({
            "frame_id": frame_id,
            "position_board_m": [x, y, z],
            "azimuth_deg": azimuth,
            "elevation_deg": elevation,
            "distance_to_board_origin_m": float(np.linalg.norm(camera_origin_board)),
        })
        tracking = frame.get("tracking", {})
        if "reprojection_error_px" in tracking:
            reprojection_errors.append(float(tracking["reprojection_error_px"]))
        if "corners" in tracking:
            tracking_corners.append(int(tracking["corners"]))

    bins_count = max(4, int(settings.get("azimuth_bins", 8)))
    covered_bins = set()
    for pose in poses:
        normalized = (pose["azimuth_deg"] + 180.0) % 360.0
        covered_bins.add(min(bins_count - 1, int(normalized / 360.0 * bins_count)))

    frame_count = len(poses)
    mean_error = (
        float(np.mean(reprojection_errors)) if reprojection_errors else None
    )
    max_error = (
        float(np.max(reprojection_errors)) if reprojection_errors else None
    )
    minimum_frames = int(settings.get("min_frames", 10))
    minimum_bins = int(settings.get("min_azimuth_bins", 4))
    warnings = []
    if frame_count < minimum_frames:
        warnings.append(
            f"Only {frame_count} valid viewpoints; recommended minimum is {minimum_frames}."
        )
    if len(covered_bins) < minimum_bins:
        warnings.append(
            f"Viewpoint coverage is limited ({len(covered_bins)}/{bins_count} azimuth sectors)."
        )
    maximum_error = float(settings.get("max_mean_reprojection_error_px", 2.0))
    if mean_error is not None and mean_error > maximum_error:
        warnings.append(
            f"Mean ChArUco reprojection error is {mean_error:.2f}px; target <= {maximum_error:.2f}px."
        )

    return {
        "valid_viewpoints": frame_count,
        "azimuth_bins_total": bins_count,
        "azimuth_bins_covered": sorted(covered_bins),
        "azimuth_coverage_percent": 100.0 * len(covered_bins) / bins_count,
        "mean_reprojection_error_px": mean_error,
        "max_reprojection_error_px": max_error,
        "mean_charuco_corners": (
            float(np.mean(tracking_corners)) if tracking_corners else None
        ),
        "poses": poses,
        "warnings": warnings,
        "status": "pass" if not warnings else "warning",
    }


def mesh_measurements(mesh: o3d.geometry.TriangleMesh) -> dict:
    """Return mesh complexity, dimensions and topology measurements."""
    aabb_extent = mesh.get_axis_aligned_bounding_box().get_extent()
    try:
        obb_extent = mesh.get_oriented_bounding_box().extent
        obb_dimensions = sorted((np.asarray(obb_extent) * 1000.0).tolist())
    except RuntimeError:
        obb_dimensions = None

    watertight = bool(mesh.is_watertight())
    volume_mm3 = None
    if watertight:
        try:
            volume_mm3 = float(abs(mesh.get_volume()) * 1_000_000_000.0)
        except RuntimeError:
            volume_mm3 = None

    components = 0
    if len(mesh.triangles):
        _, component_counts, _ = mesh.cluster_connected_triangles()
        components = len(component_counts)

    return {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "dimensions_aabb_mm": (np.asarray(aabb_extent) * 1000.0).tolist(),
        "dimensions_obb_sorted_mm": obb_dimensions,
        "surface_area_mm2": float(mesh.get_surface_area() * 1_000_000.0),
        "watertight": watertight,
        "volume_mm3": volume_mm3,
        "connected_components": components,
    }
