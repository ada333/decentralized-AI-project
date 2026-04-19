from nodes.node import Node

class Pipeline:
    nodes: list[Node]
    prompt: str
    model_path: str



    def __init__(self, nodes: list[Node], prompt: str, model_path: str):
        self.nodes = nodes
        self.prompt = prompt
        self.model_path = model_path

    def load_model(self, model_path: str):
        # load the model from the given path
        pass
    
    def assign_layers(self, total_layers: int):
        # split layers across nodes, mark first/last as special
        pass
    
    def tokenize(self, prompt: str) -> list[int]:
        # tokenize the given prompt
        pass
    
    def detokenize(self, tokens: list[int]) -> str:
        # detokenize the given tokens
        pass

    def sample(self, hidden_states: torch.Tensor) -> int:
        # sample the next token from the given hidden states
        pass

    def embed(self, token_ids: list[int]) -> torch.Tensor:
        # embed the given token ids
        pass

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # project the given hidden states to the logits
        pass
  
    def generate(self, max_new_tokens: int = 50) -> str:
        # main logic of the pipeline - assign layers, tokenize, generate the result
        model = self.load_model(self.model_path)
        self.assign_layers(model)
        token_ids = self.tokenize(self.prompt)

        hidden_states = self.embed(token_ids)
        for node in self.nodes:
            hidden_states = node.forward(hidden_states)

        logits = self.lm_head(hidden_states)
        next_token = self.sample(logits)

        if next_token == self.eos_token_id:
            return self.detokenize(token_ids)

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

        return self.detokenize(token_ids + generated)