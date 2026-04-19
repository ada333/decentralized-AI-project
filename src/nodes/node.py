import torch
import torch.nn as nn

class Node:
    layers: list[nn.Module]
    # has_embedding: bool - for now this logic is made in pipeline, add later
    # has_lm_head: bool
    kv_cache: dict[str, list[torch.Tensor]]

    def __init__(self, layers: list[nn.Module]):
        self.layers = layers
        # self.has_embedding = has_embedding
        # self.has_lm_head = has_lm_head
        self.kv_cache = {}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass