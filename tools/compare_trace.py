from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Zig and PyTorch trace summaries and report the first divergence.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--zig", type=Path, required=True)
    parser.add_argument("--tol", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ref = json.loads(args.reference.read_text(encoding="utf-8"))["nodes"]
    zig = json.loads(args.zig.read_text(encoding="utf-8"))["nodes"]

    for idx, (lhs, rhs) in enumerate(zip(ref, zig)):
        if lhs["index"] != rhs["index"] or lhs["path"] != rhs["path"] or lhs["shape"] != rhs["shape"]:
            raise SystemExit(f"node structure mismatch at {idx}: ref={lhs['path']} zig={rhs['path']}")
        for key in ("min", "max", "mean", "l2", "first"):
            if not math.isclose(lhs[key], rhs[key], rel_tol=0.0, abs_tol=args.tol):
                raise SystemExit(
                    f"first divergence: node={lhs['index']} path={lhs['path']} key={key} ref={lhs[key]} zig={rhs[key]}"
                )

    if len(ref) != len(zig):
        raise SystemExit(f"trace length mismatch: ref={len(ref)} zig={len(zig)}")

    print("no divergence within tolerance")


if __name__ == "__main__":
    main()
