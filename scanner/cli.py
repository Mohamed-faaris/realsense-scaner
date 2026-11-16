from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from scanner.board import save_board_image
from scanner.capture import capture
from scanner.decimate import decimate_mesh
from scanner.reconstruct import reconstruct
from scanner.replay import replay_bag
from scanner.visualize import visualize_mesh


def _load_config(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rs-scan",
        description="Simple D415 + ChArUco object scanner",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    board_cmd = commands.add_parser("board", help="Generate the configured ChArUco board")
    board_cmd.add_argument("--config", default="config/scanner.yaml")
    board_cmd.add_argument("--output", default="charuco-board.png")

    capture_cmd = commands.add_parser("capture", help="Capture RGB-D viewpoints")
    capture_cmd.add_argument("--config", default="config/scanner.yaml")
    capture_cmd.add_argument("--output", default="datasets/scan01")
    capture_cmd.add_argument(
        "--record-video",
        action="store_true",
        help="Save continuous MP4 plus synchronized RGB/depth/pose samples",
    )

    replay_cmd = commands.add_parser(
        "replay-bag",
        help="Re-detect ChArUco poses and extract RGB-D samples from a recorded RealSense .db3 file",
    )
    replay_cmd.add_argument("--config", default="config/scanner.yaml")
    replay_cmd.add_argument("--bag", required=True, help="Path to RealSense .db3 recording")
    replay_cmd.add_argument("--output", default="datasets/replayed01")

    reconstruct_cmd = commands.add_parser(
        "reconstruct", help="Reconstruct a mesh from a saved RGB-D session"
    )
    reconstruct_cmd.add_argument("--config", default="config/scanner.yaml")
    reconstruct_cmd.add_argument("--dataset", default="datasets/scan01")
    reconstruct_cmd.add_argument("--output", default="outputs/scan01")
    reconstruct_cmd.add_argument("--no-view", action="store_true")
    reconstruct_cmd.add_argument(
        "--no-outlier-removal",
        action="store_true",
        help="Disable configured median/statistical depth outlier rejection for this run",
    )

    visualize_cmd = commands.add_parser(
        "visualize", help="Render configured mesh angles and a combined contact sheet"
    )
    visualize_cmd.add_argument("--config", default="config/scanner.yaml")
    visualize_cmd.add_argument("--mesh", default="outputs/scan01/mesh.ply")
    visualize_cmd.add_argument("--output", default="outputs/scan01/views")

    decimate_cmd = commands.add_parser(
        "decimate", help="Reduce an existing mesh to a low-poly triangle budget"
    )
    decimate_cmd.add_argument("--config", default="config/scanner.yaml")
    decimate_cmd.add_argument("--input", default="outputs/scan01/mesh.ply")
    decimate_cmd.add_argument("--output", default="outputs/scan01/mesh_decimated.ply")
    decimate_cmd.add_argument("--target-triangles", type=int, default=None)

    args = parser.parse_args()
    config = _load_config(args.config)

    if args.command == "board":
        save_board_image(config["board"], args.output)
    elif args.command == "capture":
        capture(config, args.output, record_video=args.record_video)
    elif args.command == "replay-bag":
        replay_bag(args.bag, args.output, config)
    elif args.command == "reconstruct":
        if args.no_outlier_removal:
            config.setdefault("outlier_removal", {})["enabled"] = False
        reconstruct(args.dataset, args.output, config, show=not args.no_view)
    elif args.command == "visualize":
        visualize_mesh(args.mesh, args.output, config)
    elif args.command == "decimate":
        decimation = config["decimation"]
        target = args.target_triangles or int(decimation["target_triangles"])
        maximum_error = decimation.get("maximum_error")
        decimate_mesh(
            args.input,
            args.output,
            target_triangles=target,
            maximum_error=float("inf") if maximum_error is None else float(maximum_error),
            boundary_weight=float(decimation.get("boundary_weight", 1.0)),
        )


if __name__ == "__main__":
    main()
