import torch
import torch.nn as nn

class Node:
    layers: list[nn.Module]
    has_embedding: bool
    has_lm_head: bool
    kv_cache: dict[str, list[torch.Tensor]]

    def __init__(self, layers: list[nn.Module], has_embedding = False, has_lm_head = False):
        self.layers = layers
        self.has_embedding = has_embedding
        self.has_lm_head = has_lm_head or len(layers) == 0
        self.kv_cache = {}


    def generate(self, token: int) -> int:
        pass