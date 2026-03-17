from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyTorch node trace summaries for zero-input YOLO11s.")
    parser.add_argument("--graph", type=Path, default=Path("artifacts/graph.json"))
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("reference_trace_64.json"))
    return parser.parse_args()


def summarize(index: int, path: str, kind: str, tensor: torch.Tensor) -> dict[str, Any]:
    array = tensor.detach().cpu().float().numpy()
    return {
        "index": index,
        "path": path,
        "kind": kind,
        "shape": list(array.shape),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "l2": float(np.sqrt((array * array).sum())),
        "first": float(array.reshape(-1)[0]),
    }


def main() -> None:
    args = parse_args()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    weights = args.weights or Path(graph["source_weights"])

    wrapper = YOLO(weights.resolve().as_posix())
    model = wrapper.model.fuse().eval()

    current: Any = torch.zeros((1, 3, args.size, args.size), dtype=torch.float32)
    outputs: list[Any] = []
    nodes: list[dict[str, Any]] = []

    with torch.no_grad():
        for layer in model.model:
            if layer.f != -1:
                if isinstance(layer.f, int):
                    current = outputs[layer.f]
                else:
                    current = [current if source == -1 else outputs[source] for source in layer.f]
            current = layer(current)
            outputs.append(current)
            if isinstance(current, torch.Tensor):
                nodes.append(summarize(int(layer.i), f"model.{int(layer.i)}", layer.__class__.__name__, current))

    args.output.write_text(json.dumps({"nodes": nodes}, ensure_ascii=True, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
