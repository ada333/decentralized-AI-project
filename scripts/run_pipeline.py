"""
Run the pipeline by connecting to nodes defined in config files.

Usage:
  python -m run_pipeline --configs ../configs/node_0.toml,../configs/node_1.toml,../configs/node_2.toml --prompt "Do you like dogs?"
"""

import argparse
import asyncio
import os
import sys
import tomllib

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from node.node import Node
from pipeline.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the inference pipeline")
    parser.add_argument("--configs", type=str, required=True, help="Comma-separated paths to node config files")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate (default: 50)")
    args = parser.parse_args()

    config_paths = args.configs.split(",")

    nodes = []
    shards_dir = None
    for path in config_paths:
        with open(path.strip(), "rb") as f:
            config = tomllib.load(f)

        if shards_dir is None:
            shards_dir = config["model"]["shards_dir"]

        node = Node(
            shards_dir=config["model"]["shards_dir"],
            layer_start=config["model"]["layer_start"],
            layer_end=config["model"]["layer_end"],
            host=config["network"]["host"],
            port=config["network"]["port"],
        )
        nodes.append(node)

    nodes.sort(key=lambda n: n.layer_start)

    pipeline = Pipeline(nodes, args.prompt, shards_dir)
    result = asyncio.run(pipeline.decentralize_inference(args.max_tokens))
    print(result)


if __name__ == "__main__":
    main()
