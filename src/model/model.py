import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding


class Model:
    """Handles model-specific operations: tokenization, embedding, LM head, position embeddings."""

    def __init__(self, shards_dir: str):
        self.shards_dir = shards_dir
        self.tokenizer = None
        self.eos_token_id = None
        self._embedding = None
        self._lm_head = None
        self._norm = None
        self._rotary_emb = None
        self._hidden_size = None
        self._seq_len = 0

    def load(self):
        """Load tokenizer and pipeline head components (embedding, lm_head, norm, rotary_emb)."""
        tokenizer_dir = os.path.join(self.shards_dir, "tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.eos_token_id = self.tokenizer.eos_token_id

        head = torch.load(
            os.path.join(self.shards_dir, "pipeline_head.pt"),
            weights_only=True,
        )

        config = AutoConfig.from_pretrained(tokenizer_dir)
        self._hidden_size = config.hidden_size

        self._embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self._embedding.load_state_dict(head["embed_tokens"])

        self._lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._lm_head.load_state_dict(head["lm_head"])

        self._norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._norm.load_state_dict(head["norm"])

        self._rotary_emb = LlamaRotaryEmbedding(config=config)
        self._rotary_emb.load_state_dict(head["rotary_emb"])

    def tokenize(self, prompt: str) -> list[int]:
        """Convert text to token IDs."""
        return self.tokenizer.encode(prompt)

    def detokenize(self, tokens: list[int]) -> str:
        """Convert token IDs back to text."""
        return self.tokenizer.decode(tokens)

    def embed(self, token_ids: list[int]) -> torch.Tensor:
        """Convert token IDs to hidden states."""
        ids = torch.tensor([token_ids], dtype=torch.long)
        return self._embedding(ids)

    def apply_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply layer norm and project hidden states to vocabulary logits."""
        normed = self._norm(hidden_states)
        return self._lm_head(normed)

    def get_position_embeddings(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary position embeddings for the next seq_len tokens."""
        position_ids = torch.arange(self._seq_len, self._seq_len + seq_len).unsqueeze(0)
        dummy = torch.zeros(1, seq_len, self._hidden_size, dtype=torch.float32)
        cos, sin = self._rotary_emb(dummy, position_ids)
        self._seq_len += seq_len
        return cos, sin

    def sample(self, logits: torch.Tensor) -> int:
        """Sample the next token from logits using greedy decoding."""
        return logits[:, -1, :].argmax(dim=-1).item()

    def reset_position(self):
        """Reset position counter for a new generation session."""
        self._seq_len = 0
