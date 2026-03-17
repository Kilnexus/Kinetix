from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the fused YOLO11s model into a project-owned graph and weights format."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("deployment/traffic-signs-yolo11s/model.pt"),
        help="Path to the trained YOLO11s PyTorch checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("zig_full_runtime/artifacts"),
        help="Output directory for graph.json and weights.bin.",
    )
    return parser.parse_args()


def tensor_to_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def scalar_or_shape(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [scalar_or_shape(item) for item in value]
    if hasattr(value, "shape"):
        return {"shape": list(value.shape)}
    if value.__class__.__name__ in {"SiLU", "Identity"}:
        return value.__class__.__name__
    return str(value)


def collect_attrs(module: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key in (
        "inplace",
        "c",
        "add",
        "stride",
        "scale_factor",
        "mode",
        "padding",
        "groups",
        "reg_max",
        "nc",
        "nl",
        "no",
        "num_heads",
        "attn_ratio",
    ):
        if hasattr(module, key):
            attrs[key] = scalar_or_shape(getattr(module, key))
    if hasattr(module, "conv"):
        conv = module.conv
        attrs["conv2d"] = {
            "in_channels": conv.in_channels,
            "out_channels": conv.out_channels,
            "kernel_size": list(conv.kernel_size),
            "stride": list(conv.stride),
            "padding": list(conv.padding),
            "groups": conv.groups,
            "bias": conv.bias is not None,
        }
    if hasattr(module, "f"):
        from_value = getattr(module, "f")
        attrs["from"] = from_value if isinstance(from_value, list) else [from_value]
    if hasattr(module, "act"):
        attrs["activation"] = module.act.__class__.__name__
    return attrs


def module_to_dict(path: str, module: Any) -> dict[str, Any]:
    children = []
    for child_name, child in module.named_children():
        child_path = f"{path}.{child_name}" if path else child_name
        children.append(module_to_dict(child_path, child))

    return {
        "path": path,
        "kind": module.__class__.__name__,
        "attrs": collect_attrs(module),
        "children": children,
    }


def write_weights(model: Any, output_path: Path) -> list[dict[str, Any]]:
    offset = 0
    tensors_meta: list[dict[str, Any]] = []
    state_dict = model.state_dict()

    with output_path.open("wb") as handle:
        for name, tensor in state_dict.items():
            contiguous = tensor.detach().cpu().contiguous().float()
            raw = contiguous.numpy().tobytes(order="C")
            handle.write(raw)
            tensors_meta.append(
                {
                    "name": name,
                    "dtype": "f32",
                    "shape": list(contiguous.shape),
                    "offset": offset,
                    "nbytes": len(raw),
                }
            )
            offset += len(raw)

    return tensors_meta


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_wrapper = YOLO(args.weights.resolve().as_posix())
    model = model_wrapper.model.fuse()

    graph = {
        "format_version": 1,
        "model_name": "traffic-signs-yolo11s-full",
        "source_weights": str(args.weights.resolve()),
        "input": {"shape": [1, 3, 320, 320], "dtype": "f32"},
        "metadata": {
            "stride": tensor_to_list(model.stride),
            "save": list(model.save),
            "class_count": int(model.yaml["nc"]),
            "class_names": model.names,
        },
        "execution_plan": [
            {
                "index": int(layer.i),
                "path": f"model.{layer.i}",
                "kind": layer.__class__.__name__,
                "from": [layer.f] if isinstance(layer.f, int) else list(layer.f),
            }
            for layer in model.model
        ],
        "module_tree": module_to_dict("model", model),
    }

    weights_meta = write_weights(model, output_dir / "weights.bin")
    graph["tensors"] = weights_meta

    graph_path = output_dir / "graph.json"
    graph_path.write_text(json.dumps(graph, ensure_ascii=True, indent=2), encoding="utf-8")

    manifest = {
        "graph": graph_path.name,
        "weights": "weights.bin",
        "weights_sha_placeholder": None,
        "tensor_count": len(weights_meta),
        "weights_bytes": sum(item["nbytes"] for item in weights_meta),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    print(graph_path)
    print(output_dir / "weights.bin")


if __name__ == "__main__":
    main()
