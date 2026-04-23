"""
Start a single node from a TOML config file.

Usage:
  python -m run_node --config ../configs/node_0.toml
"""

import argparse
import asyncio
import os
import sys
import tomllib

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from node.node import Node


def main():
    parser = argparse.ArgumentParser(description="Start a node")
    parser.add_argument("--config", type=str, required=True, help="Path to node TOML config")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    node = Node(
        shards_dir=config["model"]["shards_dir"],
        layer_start=config["model"]["layer_start"],
        layer_end=config["model"]["layer_end"],
        host=config["network"]["host"],
        port=config["network"]["port"],
    )

    print(f"Starting {config['node']['name']} on {node.host}:{node.port} (layers {node.layer_start}-{node.layer_end})")
    asyncio.run(node.start())


if __name__ == "__main__":
    main()
