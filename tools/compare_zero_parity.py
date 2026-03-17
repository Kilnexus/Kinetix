from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Axionyx Zig zero-input detections against PyTorch reference.")
    parser.add_argument("--graph", type=Path, default=Path("artifacts/graph.json"))
    parser.add_argument("--weights-bin", type=Path, default=Path("artifacts/weights.bin"))
    parser.add_argument("--weights", type=Path, default=None, help="PyTorch .pt weights. Defaults to graph.source_weights.")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--zig", default="zig")
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--keep-json", action="store_true")
    parser.add_argument("--box-tol", type=float, default=1e-3)
    parser.add_argument("--score-tol", type=float, default=1e-6)
    return parser.parse_args()


def load_graph(graph_path: Path) -> dict[str, Any]:
    return json.loads(graph_path.read_text(encoding="utf-8"))


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def iou(lhs: dict[str, float], rhs: dict[str, float]) -> float:
    inter_x1 = max(lhs["x1"], rhs["x1"])
    inter_y1 = max(lhs["y1"], rhs["y1"])
    inter_x2 = min(lhs["x2"], rhs["x2"])
    inter_y2 = min(lhs["y2"], rhs["y2"])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    lhs_area = max(0.0, lhs["x2"] - lhs["x1"]) * max(0.0, lhs["y2"] - lhs["y1"])
    rhs_area = max(0.0, rhs["x2"] - rhs["x1"]) * max(0.0, rhs["y2"] - rhs["y1"])
    union_area = lhs_area + rhs_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def classwise_nms(candidates: list[dict[str, float]], iou_threshold: float, max_det: int) -> list[dict[str, float]]:
    states = [0] * len(candidates)
    selected: list[dict[str, float]] = []
    while len(selected) < max_det:
        winner = None
        best_score = -1.0
        for idx, det in enumerate(candidates):
            if states[idx] != 0:
                continue
            if det["score"] > best_score:
                best_score = det["score"]
                winner = idx
        if winner is None:
            break
        states[winner] = 2
        selected.append(candidates[winner])
        for idx, det in enumerate(candidates):
            if states[idx] != 0 or det["class_id"] != candidates[winner]["class_id"]:
                continue
            if iou(det, candidates[winner]) > iou_threshold:
                states[idx] = 1
    return selected


def reference_detections(
    weights_path: Path,
    size: int,
    score_threshold: float,
    iou_threshold: float,
    max_det: int,
) -> dict[str, Any]:
    wrapper = YOLO(weights_path.resolve().as_posix())
    model = wrapper.model.fuse().eval()

    x = torch.zeros((1, 3, size, size), dtype=torch.float32)
    with torch.no_grad():
        pred = model(x)

    if isinstance(pred, tuple):
        pred = pred[0]
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred_np = pred.detach().cpu().numpy()[0]
    boxes = xywh_to_xyxy(pred_np[:4].T)
    scores = pred_np[4:].T
    best_class = scores.argmax(axis=1)
    best_score = scores[np.arange(scores.shape[0]), best_class]

    candidates: list[dict[str, float]] = []
    for idx in range(boxes.shape[0]):
        if float(best_score[idx]) < score_threshold:
            continue
        candidates.append(
            {
                "x1": float(boxes[idx, 0]),
                "y1": float(boxes[idx, 1]),
                "x2": float(boxes[idx, 2]),
                "y2": float(boxes[idx, 3]),
                "score": float(best_score[idx]),
                "class_id": int(best_class[idx]),
            }
        )

    selected = classwise_nms(candidates, iou_threshold, max_det)
    return {"candidate_count": len(candidates), "detections": selected}


def zig_detections(
    zig: str,
    workdir: Path,
    graph_path: Path,
    weights_bin_path: Path,
    size: int,
) -> tuple[dict[str, Any], Path]:
    with tempfile.NamedTemporaryFile(prefix="axionyx-zig-", suffix=".json", delete=False) as handle:
        json_path = Path(handle.name)

    cmd = [
        zig,
        "build",
        "run",
        "--",
        graph_path.as_posix(),
        weights_bin_path.as_posix(),
        str(size),
        json_path.as_posix(),
    ]
    subprocess.run(cmd, cwd=workdir, check=True)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data, json_path


def assert_close(ref: dict[str, Any], zig: dict[str, Any], box_tol: float, score_tol: float) -> None:
    if ref["candidate_count"] != zig["candidate_count"]:
        raise AssertionError(f"candidate_count mismatch: ref={ref['candidate_count']} zig={zig['candidate_count']}")
    if len(ref["detections"]) != len(zig["detections"]):
        raise AssertionError(f"detection_count mismatch: ref={len(ref['detections'])} zig={len(zig['detections'])}")

    for idx, (lhs, rhs) in enumerate(zip(ref["detections"], zig["detections"])):
        if lhs["class_id"] != rhs["class_id"]:
            raise AssertionError(f"detection[{idx}].class_id mismatch: ref={lhs['class_id']} zig={rhs['class_id']}")
        for key in ("x1", "y1", "x2", "y2"):
            if not math.isclose(lhs[key], rhs[key], abs_tol=box_tol, rel_tol=0.0):
                raise AssertionError(f"detection[{idx}].{key} mismatch: ref={lhs[key]} zig={rhs[key]}")
        if not math.isclose(lhs["score"], rhs["score"], abs_tol=score_tol, rel_tol=0.0):
            raise AssertionError(f"detection[{idx}].score mismatch: ref={lhs['score']} zig={rhs['score']}")


def main() -> None:
    args = parse_args()
    graph = load_graph(args.graph)
    weights_path = args.weights or Path(graph["source_weights"])

    ref = reference_detections(
        weights_path=weights_path,
        size=args.size,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_det=args.max_det,
    )
    zig, zig_json_path = zig_detections(
        zig=args.zig,
        workdir=args.workdir.resolve(),
        graph_path=args.graph.resolve(),
        weights_bin_path=args.weights_bin.resolve(),
        size=args.size,
    )

    assert_close(ref, zig, box_tol=args.box_tol, score_tol=args.score_tol)

    print(f"candidate_count: {ref['candidate_count']}")
    print(f"detection_count: {len(ref['detections'])}")
    if ref["detections"]:
        print(json.dumps(ref["detections"][0], ensure_ascii=True))

    if args.keep_json:
        ref_path = args.workdir / f"reference_zero_{args.size}.json"
        zig_path = args.workdir / f"zig_zero_{args.size}.json"
        ref_path.write_text(json.dumps(ref, ensure_ascii=True, indent=2), encoding="utf-8")
        zig_path.write_text(json.dumps(zig, ensure_ascii=True, indent=2), encoding="utf-8")
    else:
        zig_json_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
