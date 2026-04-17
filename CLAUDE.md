# Decentralized AI Project

A peer-to-peer system where multiple nodes collaboratively run inference on a language model — no single machine holds the full model. Each node owns a slice of the model (a subset of layers), and token generation happens by passing activations through a pipeline of nodes. Think of it as "Petals, but as a learning project for small models."

## Project Philosophy

- **Learn by building**: Every component should be understandable. Prefer simple implementations that can be studied over opaque libraries.
- **Small models only**: Target freely available models that can run on consumer hardware. Primary target: SmolLM-135M or TinyLlama-1.1B (both Apache 2.0, downloadable from HuggingFace). The model is small enough for one machine — we split it anyway to learn the mechanics.
- **Incremental complexity**: Start with the simplest version that works. Two nodes passing activations beats an overengineered framework.
- **Make it work first**: Every network hop adds latency per token — that's fine. Correctness comes before speed. Get tokens flowing through the pipeline, even if it's slow. Latency optimization is a later phase.

## Core Idea

In normal inference, one machine runs all layers sequentially:

```
Input tokens → [Layer 0 → Layer 1 → ... → Layer N] → Logits → Sample token
                         one machine
```

In decentralized inference, the layers are split across nodes:

```
Input tokens → [Layer 0-3] → activations → [Layer 4-7] → activations → [Layer 8-11] → Logits → Sample token
                 Node A        network        Node B        network        Node C
```

Each token generation step requires a full forward pass through the pipeline. The KV cache is distributed — each node maintains the cache for its own layers.

## Architecture Overview

There are two distinct roles: the **Pipeline** (coordinator/client) and the **Nodes** (workers that hold model layers).

```
┌──────────────────────────────────────────────────────────────┐
│                       Pipeline (coordinator)                  │
│                                                               │
│  ┌────────────┐  ┌────────────┐  ┌──────────┐  ┌─────────┐  │
│  │ Tokenizer  │  │  Layer     │  │ Sampling │  │  CLI /  │  │
│  │ (encode/   │  │  Assigner  │  │ (top-k,  │  │  API    │  │
│  │  decode)   │  │            │  │  temp)   │  │         │  │
│  └─────┬──────┘  └─────┬──────┘  └────┬─────┘  └────┬────┘  │
│        │               │              │              │        │
│        └───────────────┴──────┬───────┘              │        │
│                               │                      │        │
│                    ┌──────────┴──────────┐            │        │
│                    │  Generation Loop    │◄───────────┘        │
│                    │  (drive tokens      │                     │
│                    │   through nodes)    │                     │
│                    └──────────┬──────────┘                     │
│                               │                               │
└───────────────────────────────┼───────────────────────────────┘
                                │ network
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
   ┌──────┴──────┐     ┌───────┴──────┐     ┌───────┴──────┐
   │   Node A    │     │   Node B     │     │   Node C     │
   │             │     │              │     │              │
   │ embedding + │────▶│ layers 10-19 │────▶│ layers 20-29 │
   │ layers 0-9  │     │              │     │ + LM head    │
   │             │     │              │     │              │
   │ [KV cache]  │     │ [KV cache]   │     │ [KV cache]   │
   └─────────────┘     └──────────────┘     └──────────────┘
```

**Pipeline** responsibilities:
- Holds the tokenizer — converts text → token IDs (encode) and token IDs → text (decode)
- Owns the list of nodes and knows which layers each node has
- Assigns layers to nodes (even split for V1, smarter strategies later)
- Drives the generation loop: sends token IDs to Node A, receives logits from last node, samples next token, repeats
- Nodes never see text — only token IDs (Node A) or hidden-state tensors

**Node** responsibilities:
- Holds its assigned model layers (a contiguous slice)
- First node also holds the embedding layer, last node also holds the LM head
- Runs forward pass on received input, sends output to next node (or back to Pipeline)
- Maintains its own KV cache for its layers

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python 3.11+ | Best ML ecosystem, readable, good for learning |
| ML framework | PyTorch | Transparent, imperative, easy to split models |
| Networking | asyncio + TCP (later: QUIC/libp2p) | Start simple, upgrade when needed |
| Serialization | MessagePack for control messages, raw bytes for tensors | Tensors are big — avoid encoding overhead |
| Tensor transfer | Custom binary protocol (shape header + float16 blob) | Minimal overhead for activation passing |
| Local storage | SQLite via aiosqlite | Zero setup, good enough for node state |
| CLI | Click or Typer | Clean CLI for node management |
| Testing | pytest + pytest-asyncio | Standard, well-supported |
| Config | TOML files | Human-readable, native Python 3.11+ support (tomllib) |

## Directory Structure

```
decentralized-AI-project/
├── src/
│   ├── node/              # Node lifecycle, identity, configuration
│   │   ├── __init__.py
│   │   ├── identity.py    # Keypair generation, node ID
│   │   ├── config.py      # TOML config loading
│   │   └── state.py       # Local SQLite state management
│   │
│   ├── network/           # P2P networking layer
│   │   ├── __init__.py
│   │   ├── transport.py   # TCP connection handling
│   │   ├── protocol.py    # Message types and control protocol
│   │   ├── discovery.py   # Peer discovery (bootstrap + gossip)
│   │   ├── tensor_wire.py # Efficient tensor serialization over the wire
│   │   └── router.py      # Message routing and dispatch
│   │
│   ├── ml/                # Machine learning layer
│   │   ├── __init__.py
│   │   ├── models.py      # Model architectures (full definitions)
│   │   ├── shard.py       # Model splitting — extract layer ranges
│   │   ├── forward.py     # Forward pass for a shard (partial inference)
│   │   └── kv_cache.py    # KV cache management for transformer layers
│   │
│   ├── pipeline/          # Distributed inference coordination
│   │   ├── __init__.py
│   │   ├── manager.py     # Pipeline topology — who has which layers
│   │   ├── scheduler.py   # Request routing and load balancing
│   │   ├── session.py     # Inference session state (tracks a generation)
│   │   └── sampler.py     # Token sampling (temperature, top-k, top-p)
│   │
│   ├── security/          # Security
│   │   ├── __init__.py
│   │   ├── crypto.py      # Encryption, signing, key management
│   │   └── trust.py       # Peer reputation and trust scoring
│   │
│   └── cli/               # Command-line interface
│       ├── __init__.py
│       └── main.py        # CLI commands (start, stop, status, peers, generate)
│
├── tests/
│   ├── test_network/
│   ├── test_ml/
│   ├── test_pipeline/
│   └── conftest.py        # Shared fixtures (mock nodes, test pipelines)
│
├── configs/
│   └── default.toml       # Default node configuration
│
├── scripts/
│   ├── run_local_cluster.py   # Spin up N nodes locally for testing
│   └── split_model.py        # Pre-split a model into shards for nodes
│
├── pyproject.toml
└── README.md
```

## Core Components — Design Notes

### 1. Model Sharding

The fundamental operation: take a full model and split it into contiguous layer ranges.

```python
# Conceptually (SmolLM-135M has 30 layers):
full_model = load_smollm_135m()
shard_A = full_model.layers[0:10]    # Node A
shard_B = full_model.layers[10:20]   # Node B
shard_C = full_model.layers[20:30]   # Node C

# Node A also gets the token embedding layer
# Node C also gets the final LM head (logits projection)
```

Important details:
- The **embedding layer** lives on the first node (converts token IDs → hidden states)
- The **LM head** lives on the last node (converts hidden states → logits)
- Layer norm / final norm placement depends on the architecture
- Each shard is a standalone `nn.Module` that takes hidden states in and produces hidden states out
- Shards are saved/loaded independently so nodes only download what they need

### 2. The Inference Pipeline

Token generation is **autoregressive** — the model generates one token at a time, and each new token depends on everything before it.

#### How a single model generates text (no network, one machine)

Say the prompt is `"The cat sat"`. Here's what happens inside a transformer:

1. **Tokenize**: `"The cat sat"` → token IDs `[464, 3797, 3332]`
2. **Embed**: Convert token IDs into vectors (hidden states). Each token becomes a vector of 576 numbers (for SmolLM-135M). So 3 tokens → a matrix of shape `[3, 576]`.
3. **Run through layers**: The hidden states pass through every transformer layer (30 layers in SmolLM-135M). Each layer has an **attention** step and a **feedforward** step. After all 30 layers, we have a final hidden state for each token position.
4. **LM head**: The final hidden state of the **last** token (`sat`) gets projected to vocabulary size (e.g. 49152 for SmolLM). This gives a probability score for every possible next word. These scores are called **logits**.
5. **Sample**: Pick the next token from those probabilities. Say it picks `on` (token ID 319).
6. **Repeat**: Now the sequence is `"The cat sat on"`. Run step 2-5 again to get the next token, and so on.

#### What is the KV cache and why does it matter?

The attention mechanism in each transformer layer works like this: every token looks at every previous token to decide what's important. To do this, each token produces three things:
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What do I contain?"
- **V (Value)**: "What information do I provide if selected?"

The attention score between two tokens is `Q of current token` × `K of other token`. High score means the other token is relevant. Then the output is a weighted sum of all the V vectors.

**The problem with generating token-by-token**: When we generate token 4 (`on`), it needs to attend to tokens 1-3 (`The cat sat`). To compute attention, it needs the K and V vectors for all previous tokens. We have two choices:

1. **No cache (wasteful)**: Re-run the entire sequence `[The, cat, sat, on]` through all layers from scratch. This recomputes the K and V for `The`, `cat`, `sat` even though they haven't changed. For each new token, work grows quadratically.

2. **KV cache (smart)**: After processing `[The, cat, sat]`, save the K and V vectors from every layer. When generating the next token `on`, only run `on` through the layers. At each attention step, look up the cached K and V for the previous tokens, append the new K and V for `on`, compute attention, done. The cache grows by one entry per token per layer, but we never redo old work.

```
                        KV Cache (per layer)
                        ┌──────────────────────────┐
Generating "The"     →  │ K₁, V₁                   │
Generating "cat"     →  │ K₁, V₁, K₂, V₂          │
Generating "sat"     →  │ K₁, V₁, K₂, V₂, K₃, V₃ │
Generating "on"      →  │ K₁...V₃, K₄, V₄         │  ← only K₄,V₄ are new
                        └──────────────────────────┘
Each layer has its own cache. SmolLM-135M has 30 layers → 30 separate caches.
```

#### How this works when the model is split across nodes

**Key insight: nodes never see text.** The Pipeline handles all tokenization. Only Node A receives token IDs (from the Pipeline). Nodes B, C, etc. only ever see hidden-state tensors — arrays of numbers, not words.

Concrete example — prompt `"The cat sat"`, 3 nodes, each with 10 layers:

```
STEP 1: Process the prompt (3 tokens at once)

Pipeline (coordinator):
  - Tokenizes "The cat sat" → [464, 3797, 3332]
  - Sends token IDs to Node A

Node A (layers 0-9 + embedding):
  - Embeds tokens → hidden states [3, 576]
  - Runs through layers 0-9
  - Each layer caches K, V for all 3 tokens  ← Node A's KV cache: layers 0-9
  - Sends output hidden states [3, 576] to Node B      (6.75 KB over network)

Node B (layers 10-19):
  - Receives hidden states [3, 576]
  - Runs through layers 10-19
  - Each layer caches K, V for all 3 tokens  ← Node B's KV cache: layers 10-19
  - Sends output hidden states [3, 576] to Node C

Node C (layers 20-29 + LM head):
  - Receives hidden states [3, 576]
  - Runs through layers 20-29
  - Each layer caches K, V for all 3 tokens  ← Node C's KV cache: layers 20-29
  - LM head: takes last token's hidden state → logits [49152]
  - Sends logits back to Pipeline

Pipeline:
  - Receives logits, samples next token: "on" (ID 319)
  - Decodes so far: "The cat sat on"


STEP 2: Generate next token (just 1 token this time — the cache handles history)

Pipeline:
  - Sends ONLY the new token ID [319] to Node A

Node A:
  - Embeds token → hidden state [1, 576]      (just ONE token, not the whole sequence)
  - Runs through layers 0-9
  - Attention uses CACHED K,V for "The","cat","sat" + new K,V for "on"
  - Appends new K, V to cache
  - Sends output [1, 576] to Node B             (1.15 KB — much smaller than step 1)

Node B:
  - Same process with its cache for layers 10-19
  - Sends [1, 576] to Node C

Node C:
  - Same process with its cache for layers 20-29
  - LM head → logits
  - Sends logits back to Pipeline

Pipeline:
  - Samples "the" (ID 262) from logits
  - Decodes so far: "The cat sat on the"

STEP 3, 4, 5...: Same as step 2. One token at a time, ~1.15 KB flowing through the pipeline each time.
```

**Who knows what**: The Pipeline is the only thing that touches text — it tokenizes the prompt and decodes the output. Node A receives token IDs and converts them to hidden states via the embedding layer. Nodes B, C, etc. never see text or token IDs — only hidden-state tensors. Each node remembers past context through its own slice of the KV cache. Sampling (picking the next token from logits) also happens in the Pipeline, not in the nodes.

**Activation size**: For SmolLM-135M with hidden_dim=576, one token's hidden state is 576 float16 values = 1.15 KB. During prompt processing, multiply by the number of prompt tokens. During generation, it's always just 1.15 KB per step — very manageable even on slow networks.

### 3. Pipeline Topology

How nodes organize into a pipeline:

**Phase 1 — Static assignment**: Config file says "Node A has layers 0-3, Node B has layers 4-7." Pipeline order is predetermined. Simple, works for local testing.

**Phase 2 — Dynamic assignment**: Nodes advertise their capabilities (RAM, compute speed). A coordinator assigns layer ranges based on capacity. Nodes can join/leave and the pipeline reorganizes.

**Phase 3 (stretch) — Redundancy**: Multiple nodes can hold the same layers. Requests get routed to the least-loaded replica. Provides fault tolerance.

### 4. P2P Networking

Same phased approach as any P2P system:

**Phase 1 — Direct TCP**: Nodes connect by IP:port from config. Pipeline is hardcoded.

**Phase 2 — Gossip discovery**: Nodes share peer lists and advertise which layers they hold. New nodes find the pipeline automatically.

**Phase 3 (stretch) — NAT traversal**: Hole punching, relay nodes for internet deployment.

**Two types of traffic**:
- **Control messages** (MessagePack): peer discovery, heartbeats, pipeline negotiation, session management. Small, infrequent.
- **Tensor data** (raw binary): activations flowing through the pipeline. Larger, latency-sensitive. Use a dedicated binary protocol: `[4 bytes ndim][4 bytes per dim][dtype byte][raw tensor data]`.

### 5. Node Identity

Every node needs a stable identity. Use Ed25519 keypairs.

- Node ID = hash of public key (hex-encoded, truncated to 16 chars for readability)
- Keypair stored locally in `~/.decentralized-ai/node_key`
- All control messages are signed; peers verify signatures before processing
- Tensor data between established pipeline peers can skip signing for speed (session-based trust)

### 6. Inference Sessions

A "session" tracks one generation request through the pipeline:

```python
@dataclass
class InferenceSession:
    session_id: str
    prompt_tokens: list[int]
    generated_tokens: list[int]
    pipeline: list[NodeID]        # ordered list of nodes in the pipeline
    kv_cache_positions: dict      # {node_id: num_cached_tokens}
    max_new_tokens: int
    sampling_config: SamplingConfig
    status: SessionStatus         # pending | generating | complete | failed
```

Each node in the pipeline holds its portion of the KV cache keyed by session ID. Sessions time out if idle too long (free up memory).

### 7. Configuration

Node config in TOML:
```toml
[node]
name = "my-node"
data_dir = "~/.decentralized-ai"

[network]
listen_host = "0.0.0.0"
listen_port = 8765
bootstrap_peers = ["192.168.1.10:8765", "192.168.1.11:8765"]
max_peers = 50
heartbeat_interval_sec = 30

[model]
architecture = "smollm-135m"
weights_path = "./shards/shard_0.pt"  # path to this node's shard
layer_range = [0, 4]                  # layers this node is responsible for
has_embedding = true                  # first node in pipeline
has_lm_head = false                   # last node in pipeline
dtype = "float16"

[pipeline]
role = "middle"                       # "first" | "middle" | "last"
next_node = "192.168.1.11:8765"       # where to send activations
prev_node = ""                        # where activations come from (empty = client)
max_concurrent_sessions = 4
session_timeout_sec = 300

[sampling]
temperature = 0.8
top_k = 50
top_p = 0.95
max_new_tokens = 256
```

## Key Challenges

### Latency (optimize later, not now)
Every token requires a full round trip through the pipeline. With 3 nodes and ~1ms network latency between them, that's ~2ms of network overhead per token on top of compute. On a LAN this is fine. Over the internet it will be slow — and that's okay for now. The goal is correctness first. Future optimization strategies once it works:
- **Speculative decoding**: Last node guesses ahead while activations are in flight
- **Pipeline batching**: Process multiple independent requests simultaneously so the pipeline is always busy
- **Activation compression**: Quantize activations to INT8 before sending (lossy but faster)

### Fault Tolerance
If a node in the middle of the pipeline crashes, the whole generation stalls. Options:
- **Timeout + retry**: Detect failure, restart the session from checkpoint
- **Redundant shards**: Multiple nodes hold the same layers, failover on crash
- **KV cache checkpointing**: Periodically save KV cache state so a replacement node can resume

### Memory Management
Each inference session's KV cache grows with sequence length. For SmolLM-135M with hidden_dim=576, 30 layers, and sequence length 1024: the KV cache is ~67MB per session in float16. With 4 concurrent sessions on a 3-node pipeline, each node holds ~100MB of KV cache. Need to evict old sessions and cap concurrency.

### Model Consistency
All nodes must be running the same model version with matching layer assignments. If one node has a stale shard, activations will be garbage. Need a model version handshake when establishing the pipeline.

### Heterogeneous Hardware
Not all nodes are equal. A Raspberry Pi and a gaming PC shouldn't get the same number of layers. Layer assignment should factor in:
- Available RAM (constrains shard size)
- Compute speed (constrains tokens/sec per layer)
- Network bandwidth to neighbors

## Coding Conventions

### Python Style
- Type hints on all function signatures
- Docstrings on public functions (one-liner for simple, Google-style for complex)
- `async`/`await` for all I/O operations — the networking layer is async-first
- Dataclasses or Pydantic models for structured data (messages, configs, state)
- No global mutable state; pass dependencies explicitly

### Error Handling
- Custom exception hierarchy rooted at `DecentralizedAIError`
- Network errors must be retryable — never crash on a dropped connection
- Log errors with structured context (node ID, peer ID, session ID)

### Logging
- Use `structlog` for structured logging
- Every log line includes `node_id` as bound context
- Levels: DEBUG for protocol/tensor details, INFO for lifecycle events, WARNING for degraded states, ERROR for failures

### Testing Strategy
- Unit tests for pure logic (sharding math, serialization, sampling)
- Integration tests using a local multi-node pipeline (spawn 3 nodes in-process)
- Use `pytest-asyncio` for async test functions
- Fixtures: `mock_node`, `local_pipeline`, `sample_model`, `sample_shard`

## Development Phases

### Phase 1 — Foundation
- [ ] Project scaffolding (pyproject.toml, src layout, basic CLI)
- [ ] Node identity (keypair gen, node ID derivation)
- [ ] Load a small model (SmolLM-135M from HuggingFace, or TinyLlama-1.1B)
- [ ] Model sharding: split a model into N contiguous layer ranges, save/load independently
- [ ] Single-node inference: generate tokens locally as a baseline

### Phase 2 — Two-Node Pipeline
- [ ] Basic TCP transport (connect, send, receive between two nodes)
- [ ] Tensor wire protocol (serialize/deserialize activations efficiently)
- [ ] Two-node pipeline: Node A (embedding + first layers) → Node B (last layers + LM head)
- [ ] KV cache: each node maintains cache for its layers across tokens
- [ ] End-to-end: send a prompt, get generated text back through the pipeline

### Phase 3 — Multi-Node & Discovery
- [ ] Generalize to N-node pipeline
- [ ] Peer discovery (bootstrap list → gossip with layer advertisements)
- [ ] Pipeline negotiation (nodes agree on who has which layers)
- [ ] Multiple concurrent inference sessions
- [ ] Local cluster test script (3+ nodes, full generation)

### Phase 4 — Performance
- [ ] Activation compression (float16 → int8 quantization for transfer)
- [ ] Pipeline batching (overlap compute and communication)
- [ ] Benchmarking: tokens/sec vs. number of nodes, vs. single-node baseline
- [ ] Memory management: session eviction, KV cache budgets

### Phase 5 — Robustness & Polish
- [ ] Node failure detection and pipeline recovery
- [ ] Dynamic layer reassignment when nodes join/leave
- [ ] Load balancing across heterogeneous nodes
- [ ] Simple web UI or TUI for monitoring pipeline health and generating text
- [ ] Model version handshake and consistency checks

## Key Concepts to Study

- **Pipeline Parallelism**: Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism" — though for training, the pipeline concept applies to inference
- **Petals**: Borzunov et al., "Petals: Collaborative Inference and Fine-tuning of Large Language Models" — the closest existing project to what we're building
- **KV Cache**: How transformer attention caches key/value states for autoregressive generation
- **Speculative Decoding**: Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" — guessing tokens ahead to hide latency
- **Gossip Protocols**: Demers et al., "Epidemic Algorithms for Replicated Database Maintenance" — how info spreads in P2P
- **Tensor Parallelism**: Shoeybi et al., "Megatron-LM" — splitting individual layers across devices (advanced, stretch goal)

## Common Pitfalls

- **Activation shape mismatches**: If nodes disagree on hidden_dim or dtype, activations are garbage. Always validate shapes on receive.
- **KV cache memory leaks**: Sessions that never complete will accumulate cache forever. Enforce timeouts and max session limits.
- **Autoregressive bottleneck**: You can't parallelize sequential token generation for a single request. Pipeline parallelism helps throughput (multiple requests), not single-request latency.
- **Floating point non-determinism**: Same model split differently can give slightly different results due to floating point ordering. Accept small differences.
- **Embedding/LM head placement**: Forgetting that the first node needs the embedding table and the last needs the LM head is a common sharding bug.
- **Network overhead domination**: Over the internet, network latency easily exceeds compute time for small models. This is expected — making it work correctly matters more than making it fast. Optimize after it runs.
