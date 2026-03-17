from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full zero-input parity verification for Axionyx.")
    parser.add_argument("--graph", type=Path, default=Path("artifacts/graph.json"))
    parser.add_argument("--weights-bin", type=Path, default=Path("artifacts/weights.bin"))
    parser.add_argument("--weights", type=Path, default=None, help="PyTorch .pt weights. Defaults to graph.source_weights.")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--zig", default="zig")
    parser.add_argument("--python", default="python")
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--trace-tol", type=float, default=1e-4)
    parser.add_argument("--box-tol", type=float, default=1e-3)
    parser.add_argument("--score-tol", type=float, default=1e-6)
    parser.add_argument("--keep-artifacts", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    workdir = args.workdir.resolve()
    graph_path = args.graph.resolve()
    weights_bin_path = args.weights_bin.resolve()
    weights_path = args.weights.resolve() if args.weights else None

    with tempfile.TemporaryDirectory(prefix="axionyx-parity-") as tmp_dir:
        tmp = Path(tmp_dir)
        reference_trace = tmp / f"reference_trace_{args.size}.json"
        zig_trace = tmp / f"zig_trace_{args.size}.json"
        zig_zero = tmp / f"zig_zero_{args.size}.json"

        export_cmd = [
            args.python,
            "tools/parity/export_reference_trace.py",
            "--graph",
            graph_path.as_posix(),
            "--size",
            str(args.size),
            "--output",
            reference_trace.as_posix(),
        ]
        if weights_path is not None:
            export_cmd.extend(["--weights", weights_path.as_posix()])
        run(export_cmd, workdir)

        run(
            [
                args.zig,
                "build",
                "run",
                "--",
                graph_path.as_posix(),
                weights_bin_path.as_posix(),
                str(args.size),
                zig_zero.as_posix(),
                zig_trace.as_posix(),
            ],
            workdir,
        )

        run(
            [
                args.python,
                "tools/parity/compare_trace.py",
                "--reference",
                reference_trace.as_posix(),
                "--zig",
                zig_trace.as_posix(),
                "--tol",
                str(args.trace_tol),
            ],
            workdir,
        )

        compare_cmd = [
            args.python,
            "tools/parity/compare_zero_parity.py",
            "--graph",
            graph_path.as_posix(),
            "--weights-bin",
            weights_bin_path.as_posix(),
            "--size",
            str(args.size),
            "--zig",
            args.zig,
            "--workdir",
            workdir.as_posix(),
            "--box-tol",
            str(args.box_tol),
            "--score-tol",
            str(args.score_tol),
        ]
        if weights_path is not None:
            compare_cmd.extend(["--weights", weights_path.as_posix()])
        if args.keep_artifacts:
            compare_cmd.append("--keep-json")
        run(compare_cmd, workdir)

        if args.keep_artifacts:
            shutil.copy2(reference_trace, workdir / reference_trace.name)
            shutil.copy2(zig_trace, workdir / zig_trace.name)
            shutil.copy2(zig_zero, workdir / zig_zero.name)

    print(f"parity_ok size={args.size}")


if __name__ == "__main__":
    main()
