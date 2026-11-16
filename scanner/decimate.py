from __future__ import annotations

import json
from pathlib import Path

import open3d as o3d


def decimate_mesh(
    input_path: str,
    output_path: str,
    target_triangles: int,
    maximum_error: float = float("inf"),
    boundary_weight: float = 1.0,
) -> dict:
    """Simplify an existing triangle mesh using quadric-error decimation."""
    if target_triangles < 4:
        raise ValueError("target_triangles must be at least 4")

    source = Path(input_path)
    destination = Path(output_path)
    mesh = o3d.io.read_triangle_mesh(str(source))
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise ValueError(f"Input does not contain a triangle mesh: {source}")

    original_vertices = len(mesh.vertices)
    original_triangles = len(mesh.triangles)
    if target_triangles < original_triangles:
        simplified = mesh.simplify_quadric_decimation(
            target_number_of_triangles=int(target_triangles),
            maximum_error=float(maximum_error),
            boundary_weight=float(boundary_weight),
        )
    else:
        simplified = mesh

    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_triangles()
    simplified.remove_duplicated_vertices()
    simplified.remove_unreferenced_vertices()
    simplified.compute_vertex_normals()

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(
        str(destination),
        simplified,
        write_vertex_colors=True,
        write_vertex_normals=True,
    ):
        raise OSError(f"Could not write decimated mesh: {destination}")

    result = {
        "input": str(source),
        "output": str(destination),
        "input_vertices": original_vertices,
        "input_triangles": original_triangles,
        "output_vertices": len(simplified.vertices),
        "output_triangles": len(simplified.triangles),
        "triangle_reduction_percent": (
            100.0 * (1.0 - len(simplified.triangles) / original_triangles)
        ),
    }
    report_path = destination.with_suffix(".json")
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"Decimated {original_triangles} -> {len(simplified.triangles)} triangles "
        f"({result['triangle_reduction_percent']:.1f}% reduction)"
    )
    print(f"Mesh: {destination.resolve()}")
    print(f"Report: {report_path.resolve()}")
    return result
