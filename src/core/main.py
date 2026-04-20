from pipeline.pipeline import Pipeline
from node.node import Node

MODEL_PATH = "./models/smollm-135m"

def main():
    nodes = [Node() for _ in range(3)]
    pipeline = Pipeline(MODEL_PATH)

if __name__ == "__main__":
    main()