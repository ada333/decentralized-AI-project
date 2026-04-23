import asyncio
import os
import struct
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding
from node.node import Node


class Pipeline:
    nodes: list[Node]
    prompt: str
    shards_dir: str

    def __init__(self, nodes: list[Node], prompt: str, shards_dir: str):
        self.nodes = nodes
        self.prompt = prompt
        self.shards_dir = shards_dir
        self.tokenizer = None
        self.eos_token_id = None
        self._embedding = None
        self._lm_head = None
        self._norm = None
        self._rotary_emb = None
        self._seq_len = 0
        self._connections: list[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = []

    def _load_model(self):
        """Load only the pipeline head (embedding, lm_head, norm, rotary_emb) and tokenizer."""
        tokenizer_dir = os.path.join(self.shards_dir, "tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.eos_token_id = self.tokenizer.eos_token_id

        head = torch.load(
            os.path.join(self.shards_dir, "pipeline_head.pt"),
            weights_only=True,
        )

        config = AutoConfig.from_pretrained(tokenizer_dir)

        self._embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self._embedding.load_state_dict(head["embed_tokens"])

        self._lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._lm_head.load_state_dict(head["lm_head"])

        self._norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._norm.load_state_dict(head["norm"])

        self._rotary_emb = LlamaRotaryEmbedding(config=config)
        self._rotary_emb.load_state_dict(head["rotary_emb"])

    async def _connect_to_nodes(self):
        """Open a TCP connection to each node."""
        for node in self.nodes:
            reader, writer = await asyncio.open_connection(node.host, node.port)
            self._connections.append((reader, writer))

    async def _send(self, node_index: int, data: bytes):
        """Send length-prefixed data to a node."""
        _, writer = self._connections[node_index]
        header = struct.pack(">I", len(data))
        writer.write(header + data)
        await writer.drain()

    async def _receive(self, node_index: int) -> bytes:
        """Receive length-prefixed data from a node."""
        reader, _ = self._connections[node_index]
        header = await reader.readexactly(4)
        length = struct.unpack(">I", header)[0]
        return await reader.readexactly(length)
    
    def _tokenize(self, prompt: str) -> list[int]:
        # tokenize the given prompt into a list of token ids
        return self.tokenizer.encode(prompt)

    def _detokenize(self, tokens: list[int]) -> str:
        # detokenize the given tokens into a string
        return self.tokenizer.decode(tokens)

    def _sample(self, logits: torch.Tensor) -> int:
        # sample the next token from the given logits (probability distribution over the vocabulary)
        return logits[:, -1, :].argmax(dim=-1).item()

    def _embed(self, token_ids: list[int]) -> torch.Tensor:
        # embed the given token ids into hidden states
        ids = torch.tensor([token_ids], dtype=torch.long)
        return self._embedding(ids)

    def _apply_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self._norm(hidden_states)
        return self._lm_head(normed)
  
    def _get_position_embeddings(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary position embeddings (Llama-specific: RoPE)."""
        position_ids = torch.arange(self._seq_len, self._seq_len + seq_len).unsqueeze(0)
        hidden_size = self._embedding.embedding_dim
        dummy = torch.zeros(1, seq_len, hidden_size, dtype=torch.float32)
        cos, sin = self._rotary_emb(dummy, position_ids)
        self._seq_len += seq_len
        return cos, sin

    @torch.no_grad()
    async def decentralize_inference(self, max_new_tokens: int = 50) -> str:
        self._load_model()
        await self._connect_to_nodes()

        token_ids = self._tokenize(self.prompt)

        hidden_states = self._embed(token_ids)
        position_embeddings = self._get_position_embeddings(len(token_ids))
        next_token = await self.generate_next_token(hidden_states, position_embeddings)

        if next_token == self.eos_token_id:
            return ""

        generated = [next_token]

        for _ in range(max_new_tokens - 1):
            hidden_states = self._embed([next_token])
            position_embeddings = self._get_position_embeddings(1)
            next_token = await self.generate_next_token(hidden_states, position_embeddings)

            if next_token == self.eos_token_id:
                break

            generated.append(next_token)

        return self._detokenize(generated)

    async def generate_next_token(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> int:
        data = self._serialize_tensors(hidden_states, position_embeddings)

        for i in range(len(self.nodes)):
            await self._send(i, data)
            data = await self._receive(i)

        hidden_states, _ = self._deserialize_tensors(data)
        logits = self._apply_lm_head(hidden_states)
        return self._sample(logits)

    def _serialize_tensors(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> bytes:
        """Pack hidden states and position embeddings into bytes."""
        import io
        buffer = io.BytesIO()
        torch.save((hidden_states, position_embeddings), buffer)
        return buffer.getvalue()

    def _deserialize_tensors(self, data: bytes) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Unpack hidden states and position embeddings from bytes."""
        import io
        buffer = io.BytesIO(data)
        return torch.load(buffer, weights_only=True)