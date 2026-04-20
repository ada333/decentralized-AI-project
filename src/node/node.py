import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

class Node:
    layers: list[nn.Module]
    kv_cache: DynamicCache

    def __init__(self, layers: list[nn.Module]):
        self.layers = layers
        self.kv_cache = DynamicCache()

    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                past_key_values=self.kv_cache,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
        return hidden_states