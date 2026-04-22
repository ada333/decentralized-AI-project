import asyncio
import tomllib
import os
from pipeline.pipeline import Pipeline
from node.node import Node

SHARDS_DIR = "./models/smollm-135m-shards"
NUM_NODES = 3


def assign_layers(num_layers: int, num_nodes: int) -> list[tuple[int, int]]:
    """Distribute layers evenly across nodes. Last node gets the remainder."""
    layers_per_node = num_layers // num_nodes
    assignments = []

    start = 0
    for i in range(num_nodes):
        end = start + layers_per_node
        if i == num_nodes - 1:
            end = num_layers
        assignments.append((start, end))
        start = end

    return assignments


def main():
    with open(os.path.join(SHARDS_DIR, "model_info.toml"), "rb") as f:
        model_info = tomllib.load(f)
    num_layers = model_info["num_layers"]

    assignments = assign_layers(num_layers, NUM_NODES)

    nodes = []
    for i, (layer_start, layer_end) in enumerate(assignments):
        port = 8765 + i
        nodes.append(Node(SHARDS_DIR, layer_start, layer_end, "127.0.0.1", port))

    prompt = "Do you like dogs?"
    pipeline = Pipeline(nodes, prompt, SHARDS_DIR)
    print(asyncio.run(pipeline.decentralize_inference()))


if __name__ == "__main__":
    main()
