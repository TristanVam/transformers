"""Utilities for visualising attention heads stored during training."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch


def load_attention(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if tensor.dim() == 4:
        # Multi-head attention returns (batch, heads, query, key)
        return tensor.mean(dim=0)
    return tensor


def plot_attention(attention: torch.Tensor, head: Optional[int] = None, output: Optional[Path] = None) -> None:
    if attention.dim() == 3 and head is not None:
        matrix = attention[head]
    elif attention.dim() == 3:
        matrix = attention.mean(dim=0)
    else:
        matrix = attention

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="attention weight")
    plt.title("Sentiment cross-attention map")
    plt.xlabel("Sentiment tokens")
    plt.ylabel("Time tokens")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise saved attention heatmaps")
    parser.add_argument("attention_path", type=Path, help="Path to attention .pt file")
    parser.add_argument("--head", type=int, default=None, help="Specific head to visualise")
    parser.add_argument("--output", type=Path, default=None, help="Optional output image path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attention = load_attention(args.attention_path)
    plot_attention(attention, head=args.head, output=args.output)


if __name__ == "__main__":
    main()
