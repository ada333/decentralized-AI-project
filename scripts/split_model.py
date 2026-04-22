"""
Split a HuggingFace model into individual layer files for distributed inference.

Run once per model. Produces:
  - layer_0.pt ... layer_N.pt   (one file per transformer block)
  - pipeline_head.pt            (embedding + lm_head + norm + rotary_emb)
  - tokenizer/                  (tokenizer files)
  - model_info.toml             (metadata)

Usage:
  python scripts/split_model.py --model HuggingFaceTB/SmolLM-135M --output ./models/smollm-135m-shards
  python scripts/split_model.py --model ./models/smollm-135m --output ./models/smollm-135m-shards
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def split_model(model_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    layers = model.model.layers
    num_layers = len(layers)
    hidden_dim = model.config.hidden_size
    architecture = model.config.model_type

    print(f"Model: {architecture}, {num_layers} layers, hidden_dim={hidden_dim}")

    # Save each layer individually
    for i, layer in enumerate(layers):
        layer_path = os.path.join(output_dir, f"layer_{i}.pt")
        torch.save(layer.state_dict(), layer_path)
        print(f"  Saved layer {i}/{num_layers - 1} -> {layer_path}")

    # Save pipeline head (everything the Pipeline needs)
    head = {
        "embed_tokens": model.model.embed_tokens.state_dict(),
        "lm_head": model.lm_head.state_dict(),
        "norm": model.model.norm.state_dict(),
        "rotary_emb": model.model.rotary_emb.state_dict(),
    }
    head_path = os.path.join(output_dir, "pipeline_head.pt")
    torch.save(head, head_path)
    print(f"  Saved pipeline head -> {head_path}")

    # Save tokenizer and model config
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_dir)
    model.config.save_pretrained(tokenizer_dir)
    print(f"  Saved tokenizer + config -> {tokenizer_dir}")

    # Save model metadata
    info_path = os.path.join(output_dir, "model_info.toml")
    with open(info_path, "w") as f:
        f.write(f'architecture = "{architecture}"\n')
        f.write(f"num_layers = {num_layers}\n")
        f.write(f"hidden_dim = {hidden_dim}\n")
        f.write(f"vocab_size = {model.config.vocab_size}\n")
        f.write(f"num_heads = {model.config.num_attention_heads}\n")
    print(f"  Saved model info -> {info_path}")

    print(f"\nDone. {num_layers} layer files + pipeline_head + tokenizer saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a model into per-layer shards")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or local path")
    parser.add_argument("--output", type=str, required=True, help="Output directory for shard files")
    args = parser.parse_args()

    split_model(args.model, args.output)
