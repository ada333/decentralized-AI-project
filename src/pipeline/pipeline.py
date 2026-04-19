import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from nodes.node import Node

class Pipeline:
    nodes: list[Node]
    prompt: str
    model_path: str

    def __init__(self, nodes: list[Node], prompt: str, model_path: str):
        self.nodes = nodes
        self.prompt = prompt
        self.model_path = model_path
        self.tokenizer = None
        self.eos_token_id = None
        self._embedding = None
        self._lm_head = None
        self._norm = None

    def load_model(self, model_path: str):
        # load the model from the given path and return the layers
        # also assign the tokenizer, eos token id, embedding layer, lm head and norm layer
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)  # generic HuggingFace
        model.eval()                                                                         # generic PyTorch

        tokenizer = AutoTokenizer.from_pretrained(model_path)  # generic HuggingFace

        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self._embedding = model.model.embed_tokens  # Llama-specific: in GPT-2 this would be model.transformer.wte
        self._lm_head = model.lm_head               # Llama-specific name, but most models have this at model.lm_head
        self._norm = model.model.norm                # Llama-specific: standalone RMSNorm before lm_head, not all architectures have this

        return list(model.model.layers)              # Llama-specific: in GPT-2 this would be model.transformer.h

    def assign_layers(self, layers: list[nn.Module]):
        # assign layers to nodes equally, 
        # the last node gets the remaining layers (remainder from division)
        layers_per_node = len(layers) // len(self.nodes)

        start = 0
        for node in self.nodes:
            node.layers = list(layers[start:start + layers_per_node])
            start += layers_per_node

        self.nodes[-1].layers += layers[start:]
    
    def tokenize(self, prompt: str) -> list[int]:
        # tokenize the given prompt into a list of token ids
        return self.tokenizer.encode(prompt)

    def detokenize(self, tokens: list[int]) -> str:
        # detokenize the given tokens into a string
        return self.tokenizer.decode(tokens)

    def sample(self, logits: torch.Tensor) -> int:
        # sample the next token from the given logits (probability distribution over the vocabulary)
        return logits[:, -1, :].argmax(dim=-1).item()

    def embed(self, token_ids: list[int]) -> torch.Tensor:
        # embed the given token ids into hidden states
        ids = torch.tensor([token_ids], dtype=torch.long)
        return self._embedding(ids)

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # project the given hidden states to the logits (logits = probability distribution over the vocabulary)
        normed = self._norm(hidden_states) # Llama specific - final layer norm applied before lm_head
        return self._lm_head(normed)
  
    @torch.no_grad()
    def generate(self, max_new_tokens: int = 50) -> str:
        # main logic of the pipeline - assign layers, tokenize, generate the result
        layers = self.load_model(self.model_path)
        self.assign_layers(layers)
        token_ids = self.tokenize(self.prompt)

        hidden_states = self.embed(token_ids)
        for node in self.nodes:
            hidden_states = node.forward(hidden_states)

        logits = self.lm_head(hidden_states)
        next_token = self.sample(logits)

        if next_token == self.eos_token_id:
            return ""

        generated = [next_token]

        for _ in range(max_new_tokens - 1):
            hidden_states = self.embed([next_token])
            for node in self.nodes:
                hidden_states = node.forward(hidden_states)
            logits = self.lm_head(hidden_states)
            next_token = self.sample(logits)

            if next_token == self.eos_token_id:
                break

            generated.append(next_token)

        return self.detokenize(generated)