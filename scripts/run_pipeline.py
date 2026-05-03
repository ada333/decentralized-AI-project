"""
Run the pipeline by connecting to nodes defined in config files.

Usage:
  python scripts/run_pipeline.py --configs configs/node_0.toml,configs/node_1.toml,configs/node_2.toml
"""

import argparse
import asyncio
import os
import sys
import tomllib

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model.model import Model
from node.coordinator import PipelineCoordinator
from node.layer_groups import LAYER_GROUPS
from node.node_group import NodeInfo
from pipeline.pipeline import Pipeline


def layer_range_to_group_id(layer_range: tuple[int, int]) -> int:
    """Convert a layer range tuple to a group_id.

    Args:
        layer_range: Tuple of (start_layer, end_layer).

    Returns:
        The group_id that matches this layer range.

    Raises:
        ValueError: If no matching group is found.
    """
    for group_id, config in LAYER_GROUPS.items():
        if (config.start_layer, config.end_layer) == layer_range:
            return group_id
    raise ValueError(
        f"No matching group for layer_range {layer_range}. "
        f"Valid ranges: {[(c.start_layer, c.end_layer) for c in LAYER_GROUPS.values()]}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run the inference pipeline")
    parser.add_argument(
        "--configs",
        type=str,
        required=True,
        help="Comma-separated paths to node config files",
    )
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    args = parser.parse_args()

    config_paths = args.configs.split(",")

    coordinator = PipelineCoordinator()
    shards_dir = None

    for path in config_paths:
        with open(path.strip(), "rb") as f:
            config = tomllib.load(f)

        if shards_dir is None:
            shards_dir = config["model"]["shards_dir"]

        # Create NodeInfo from config
        layer_range = tuple(config["model"]["layer_range"])
        group_id = layer_range_to_group_id(layer_range)
        node_info = NodeInfo(
            node_id=config["node"]["name"],
            host=config["network"]["host"],
            port=config["network"]["port"],
            group_id=group_id,
        )
        coordinator.register_node(node_info)

    model = Model(shards_dir)
    model.load()

    pipeline = Pipeline(model, coordinator)

    print("Pipeline ready. Type your prompt and press Enter. Type 'exit' or Ctrl+C to quit.")
    print()

    try:
        while True:
            try:
                prompt = input(">>> ").strip()
            except EOFError:
                break

            if not prompt:
                continue

            if prompt.lower() == "exit":
                break

            result = asyncio.run(pipeline.generate(prompt, args.max_tokens))
            print(result)
            print()

    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
