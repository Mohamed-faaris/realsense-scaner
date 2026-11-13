from __future__ import annotations

import argparse
import yaml

from scanner.board import save_board_image
from scanner.capture import capture
from scanner.reconstruct import reconstruct


def main() -> None:
    parser = argparse.ArgumentParser(prog="rs-scan", description="Simple D415 object scanner")
    commands = parser.add_subparsers(dest="command", required=True)

    board_cmd = commands.add_parser("board", help="Generate your configured ChArUco board")
    board_cmd.add_argument("--config", default="config/scanner.yaml")
    board_cmd.add_argument("--output", default="charuco-board.png")

    capture_cmd = commands.add_parser("capture", help="Capture RGB-D views")
    capture_cmd.add_argument("--config", default="config/scanner.yaml")
    capture_cmd.add_argument("--output", default="datasets/scan01")

    reconstruct_cmd = commands.add_parser("reconstruct", help="Build a low-poly mesh")
    reconstruct_cmd.add_argument("--config", default="config/scanner.yaml")
    reconstruct_cmd.add_argument("--dataset", default="datasets/scan01")
    reconstruct_cmd.add_argument("--output", default="outputs/scan01")
    reconstruct_cmd.add_argument("--no-view", action="store_true")

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, encoding="utf-8")) if hasattr(args, "config") else None

    if args.command == "board":
        save_board_image(config["board"], args.output)
    elif args.command == "capture":
        capture(config, args.output)
    elif args.command == "reconstruct":
        reconstruct(args.dataset, args.output, config, show=not args.no_view)


if __name__ == "__main__":
    main()
