"""
Generate TOML config files for N nodes.

Reads model_info.toml to determine the number of layers, distributes them
evenly across nodes, and writes one config file per node to the output directory.

Usage:
  python scripts/generate_node_configs.py
"""

# THIS SCRIPT IS USED IN THE INTEGRATION TESTS - ANY CHANGE CAN AFFECT THE TESTS

import os
import tomllib

DEFAULT_SHARDS_DIR = "./models/smollm-135m-shards"
DEFAULT_NUM_NODES = 3
DEFAULT_OUTPUT_DIR = "./configs"
DEFAULT_BASE_PORT = 8765
DEFAULT_HOST = "127.0.0.1"


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


def generate_configs(
    shards_dir: str = DEFAULT_SHARDS_DIR,
    num_nodes: int = DEFAULT_NUM_NODES,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    base_port: int = DEFAULT_BASE_PORT,
    host: str = DEFAULT_HOST,
) -> list[str]:
    """Generate config files and return list of config paths."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(shards_dir, "model_info.toml"), "rb") as f:
        model_info = tomllib.load(f)
    num_layers = model_info["num_layers"]

    assignments = assign_layers(num_layers, num_nodes)
    config_paths = []

    for i, (layer_start, layer_end) in enumerate(assignments):
        name = f"node_{i}"
        port = base_port + i
        config_path = os.path.join(output_dir, f"{name}.toml")

        with open(config_path, "w") as f:
            f.write("[node]\n")
            f.write(f'name = "{name}"\n')
            f.write("\n")
            f.write("[network]\n")
            f.write(f'host = "{host}"\n')
            f.write(f"port = {port}\n")
            f.write("\n")
            f.write("[model]\n")
            f.write(f'shards_dir = "{os.path.abspath(shards_dir)}"\n')
            f.write(f"layer_start = {layer_start}\n")
            f.write(f"layer_end = {layer_end}\n")

        config_paths.append(config_path)
        print(f"  {name}: layers [{layer_start}, {layer_end}) on {host}:{port}")

    return config_paths


if __name__ == "__main__":
    configs = generate_configs()
    print(f"\nGenerated {len(configs)} config files in {DEFAULT_OUTPUT_DIR}")
