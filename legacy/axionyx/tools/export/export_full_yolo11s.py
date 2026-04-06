from __future__ import annotations

from export_ultralytics_yolo import main as generic_main


def main() -> None:
    generic_main(
        [
            "--weights",
            "E:/07_AI/ChineseTrafficSigns/deployment/traffic-signs-yolo11s/model.pt",
            "--output-dir",
            "artifacts",
            "--model-name",
            "traffic-signs-yolo11s-full",
            "--input-size",
            "320",
        ]
    )


if __name__ == "__main__":
    main()
