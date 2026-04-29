"""
Generate TOML config files for N nodes.

Reads model_info.toml to determine the number of layers, distributes them
evenly across nodes, and writes one config file per node to the output directory.

Usage:
  python scripts/generate_node_configs.py
"""

import os
import tomllib

SHARDS_DIR = "./models/smollm-135m-shards"
NUM_NODES = 1
OUTPUT_DIR = "./configs"
BASE_PORT = 8765

# For local testing (all on one machine):
HOSTS = ["192.168.0.124"] * NUM_NODES

# For multi-device testing, replace with actual IPs:
# HOSTS = [
#     "192.168.1.100",  # Node 0 - your machine
#     "192.168.1.101",  # Node 1 - second device
#     "192.168.1.102",  # Node 2 - third device
# ]


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


def generate_configs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(SHARDS_DIR, "model_info.toml"), "rb") as f:
        model_info = tomllib.load(f)
    num_layers = model_info["num_layers"]

    assignments = assign_layers(num_layers, NUM_NODES)

    for i, (layer_start, layer_end) in enumerate(assignments):
        name = f"node_{i}"
        host = HOSTS[i]
        port = BASE_PORT + i
        config_path = os.path.join(OUTPUT_DIR, f"{name}.toml")

        with open(config_path, "w") as f:
            f.write("[node]\n")
            f.write(f'name = "{name}"\n')
            f.write("\n")
            f.write("[network]\n")
            f.write(f'host = "{host}"\n')
            f.write(f"port = {port}\n")
            f.write("\n")
            f.write("[model]\n")
            f.write(f'shards_dir = "{SHARDS_DIR}"\n')
            f.write(f"layer_start = {layer_start}\n")
            f.write(f"layer_end = {layer_end}\n")

        print(f"  {name}: layers [{layer_start}, {layer_end}) on {host}:{port} -> {config_path}")

    print(f"\nGenerated {NUM_NODES} config files in {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_configs()
