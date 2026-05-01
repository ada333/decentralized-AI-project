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
from contextlib import contextmanager
from dataclasses import dataclass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from generate_node_configs import generate_configs  # noqa: E402
from model.model import Model  # noqa: E402
from pipeline.pipeline import Pipeline  # noqa: E402

SHARDS_DIR = os.path.join(PROJECT_ROOT, "models", "smollm-135m-shards")
NUM_NODES = 3
MAX_TOKENS = 15
BASE_PORT = 18765


@dataclass
class GenerationResult:
    prompt: str
    output: str
    token_count: int
    elapsed: float

    @property
    def tokens_per_sec(self) -> float:
        return self.token_count / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def success(self) -> bool:
        return bool(self.output.strip())


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


def check_shards_exist() -> bool:
    """Check if model shards are available."""
    if not os.path.exists(os.path.join(SHARDS_DIR, "model_info.toml")):
        print(f"ERROR: Model shards not found at {SHARDS_DIR}")
        print(
            "Run: python scripts/split_model.py --model HuggingFaceTB/SmolLM-135M --output",
            SHARDS_DIR,
        )
        return False
    return True


@contextmanager
def local_cluster(num_nodes: int = NUM_NODES, base_port: int = BASE_PORT):
    """Context manager that starts a local cluster and tears it down on exit."""
    temp_dir = tempfile.mkdtemp(prefix="decentralized_ai_test_")
    # this is dependant on the generate_configs script - if the script changes, this test can break
    config_paths = generate_configs(
        shards_dir=SHARDS_DIR,
        num_nodes=num_nodes,
        output_dir=temp_dir,
        base_port=base_port,
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
        for i in range(num_nodes):
            if not wait_for_port(base_port + i):
                raise RuntimeError(f"Node {i} failed to start")
        print("All nodes ready\n")

        yield

    finally:
        for proc in processes:
            proc.terminate()
            proc.wait(timeout=5)


async def run_generation(prompt: str, max_tokens: int = MAX_TOKENS) -> tuple[str, int]:
    """Run generation through the pipeline. Returns (text, token_count)."""
    model = Model(SHARDS_DIR)
    model.load()
    nodes = [("127.0.0.1", BASE_PORT + i) for i in range(NUM_NODES)]
    pipeline = Pipeline(model, nodes)
    result = await pipeline.generate(prompt, max_tokens)
    token_count = len(model.tokenize(result)) if result else 0
    return result, token_count


def print_results(results: list[GenerationResult], title: str):
    """Print generation results in a consistent format."""
    print(f"=== {title} ===")

    total_tokens = sum(r.token_count for r in results)
    total_elapsed = max(r.elapsed for r in results)

    if len(results) == 1:
        r = results[0]
        print(f'Prompt: "{r.prompt}"')
        print(f"Generated: {r.output}\n")
        print(f"Nodes: {NUM_NODES}")
        print(f"Tokens: {r.token_count}")
    else:
        for i, r in enumerate(results):
            print(f'Pipeline {i + 1}: "{r.prompt}" -> {r.output}')
        print(f"\nPipelines: {len(results)}")
        print(f"Total tokens: {total_tokens}")

    print(f"Time: {total_elapsed:.2f}s")
    if total_elapsed > 0 and total_tokens > 0:
        print(f"Speed: {total_tokens / total_elapsed:.1f} tokens/s")

    if all(r.success for r in results):
        print("\nSUCCESS")
    else:
        print("\nWARNING: Some outputs were empty")


def test_single_generation():
    """Test single prompt generation through the pipeline."""
    prompt = "Lets go for that!"

    start = time.perf_counter()
    output, token_count = asyncio.run(run_generation(prompt))
    elapsed = time.perf_counter() - start

    result = GenerationResult(prompt, output, token_count, elapsed)
    print_results([result], "Single Generation Test")
    return result.success


def test_concurrent_generation():
    """Test multiple concurrent generations through the pipeline."""
    prompts = ["Ja lasko zastavim cas", "When you call my name"]

    async def run_all():
        tasks = [run_generation(p) for p in prompts]
        return await asyncio.gather(*tasks)

    start = time.perf_counter()
    outputs = asyncio.run(run_all())
    elapsed = time.perf_counter() - start

    results = [
        GenerationResult(prompt, output, token_count, elapsed)
        for prompt, (output, token_count) in zip(prompts, outputs)
    ]

    print_results(results, "Concurrent Pipelines Test")
    return all(r.success for r in results)


def main():
    print(f"=== Local Cluster Test ({NUM_NODES} nodes) ===\n")

    if not check_shards_exist():
        sys.exit(1)

    with local_cluster():
        success = test_single_generation()

        print("\n" + "=" * 50 + "\n")

        success &= test_concurrent_generation()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
