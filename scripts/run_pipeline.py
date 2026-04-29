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
from pipeline.pipeline import Pipeline


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

    nodes_addresses = []
    shards_dir = None
    for path in config_paths:
        with open(path.strip(), "rb") as f:
            config = tomllib.load(f)

        if shards_dir is None:
            shards_dir = config["model"]["shards_dir"]
        nodes_addresses.append((config["network"]["host"], config["network"]["port"]))

    model = Model(shards_dir)
    model.load()

    pipeline = Pipeline(model, nodes_addresses)

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
