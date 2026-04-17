from nodes.node import Node

class Pipeline:
    nodes: list[Node]


    def __init__(self, nodes: list[Node]):
        self.nodes = nodes


    def load_model(self, model_path: str):
        # load the model from the given path
        pass
    
    def assign_layers(self, total_layers: int):
        # split layers across nodes, mark first/last as special
        pass
    
    def tokenize(self, text: str) -> list[int]:
        # tokenize the given text
        pass
    
    def detokenize(self, tokens: list[int]) -> str:
        # detokenize the given tokens
        pass
    
    def generate(self, prompt_tokens: list[int]) -> list[int]:
        # orchestrate the token-by-token flow through self.nodes
        pass