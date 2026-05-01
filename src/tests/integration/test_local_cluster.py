#!/usr/bin/env python3

"""
Spin up a local 3-node cluster and verify end-to-end generation works.

Usage:
  python src/tests/integration/test_local_cluster.py

this test cant be run in CI because it requires a local model to be split
and the model is not included in the repository.

To run this test, you need to:
1. Split the model using the scripts/split_model.py script
2. Run the test using the python src/tests/integration/test_local_cluster.py command

This test will spin up a local 3-node cluster and verify end-to-end generation works.

"""

import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import time

from generate_node_configs import generate_configs
from model.model import Model
from pipeline.pipeline import Pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))


SHARDS_DIR = os.path.join(PROJECT_ROOT, "models", "smollm-135m-shards")
NUM_NODES = 3
PROMPT = "Lets go for that!"
MAX_TOKENS = 15
BASE_PORT = 18765


def wait_for_port(port: int, timeout: float = 60.0) -> bool:
    """Wait until port is accepting connections."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return True
        except (TimeoutError, ConnectionRefusedError, OSError):
            time.sleep(0.2)
    return False


async def run_generation() -> tuple[str, int]:
    """Run generation through the pipeline. Returns (text, token_count)."""
    model = Model(SHARDS_DIR)
    model.load()
    nodes = [("127.0.0.1", BASE_PORT + i) for i in range(NUM_NODES)]
    pipeline = Pipeline(model, nodes)
    result = await pipeline.generate(PROMPT, MAX_TOKENS)
    token_count = len(model.tokenize(result)) if result else 0
    return result, token_count


def main():
    print(f"=== Local Cluster Test ({NUM_NODES} nodes) ===\n")

    if not os.path.exists(os.path.join(SHARDS_DIR, "model_info.toml")):
        print(f"ERROR: Model shards not found at {SHARDS_DIR}")
        print(
            "Run: python scripts/split_model.py --model HuggingFaceTB/SmolLM-135M --output",
            SHARDS_DIR,
        )
        sys.exit(1)

    temp_dir = tempfile.mkdtemp(prefix="decentralized_ai_test_")
    config_paths = generate_configs(
        shards_dir=SHARDS_DIR,
        num_nodes=NUM_NODES,
        output_dir=temp_dir,
        base_port=BASE_PORT,
    )

    run_node = os.path.join(PROJECT_ROOT, "scripts", "run_node.py")
    processes = []
    try:
        for config in config_paths:
            proc = subprocess.Popen(
                [sys.executable, run_node, "--config", config],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
            )
            processes.append(proc)

        print("\nWaiting for nodes...")
        for i in range(NUM_NODES):
            if not wait_for_port(BASE_PORT + i):
                print(f"ERROR: Node {i} failed to start")
                sys.exit(1)
        print("All nodes ready\n")

        print(f'Prompt: "{PROMPT}"')
        start = time.perf_counter()
        result, token_count = asyncio.run(run_generation())
        elapsed = time.perf_counter() - start

        print(f"Generated: {result}\n")
        print("=== Results ===")
        print(f"Nodes: {NUM_NODES}")
        print(f"Tokens: {token_count}")
        print(f"Time: {elapsed:.2f}s")
        if elapsed > 0 and token_count > 0:
            print(f"Speed: {token_count / elapsed:.1f} tokens/s")

        if result.strip():
            print("\nSUCCESS")
        else:
            print("\nWARNING: Empty output (EOS token)")

    finally:
        for proc in processes:
            proc.terminate()
            proc.wait(timeout=5)


if __name__ == "__main__":
    main()
