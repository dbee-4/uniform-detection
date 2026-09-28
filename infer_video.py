#!/usr/bin/env python3
"""
Run YOLO detection on a video file.

Classes: No-Uniform, People, Security, Uniform
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.nn.autobackend import check_class_names

CLASS_NAMES = ["No-Uniform", "People", "Security", "Uniform"]


def setup_model(
    model: YOLO, class_names: list[str], weights: Path
) -> tuple[dict[int, str], list[int]]:
    """Attach class names when the checkpoint is a true N-class detector."""
    checkpoint_nc = len(model.names)
    class_ids = list(range(len(class_names)))

    if checkpoint_nc != len(class_names):
        sample = ", ".join(f"{i}={model.names[i]}" for i in range(min(4, checkpoint_nc)))
        raise SystemExit(
            f"\nWeights '{weights.name}' has {checkpoint_nc} classes ({sample}, ...), "
            f"but this project expects {len(class_names)}: {class_names}.\n\n"
            "Use your trained checkpoint, e.g.:\n"
            f"  python infer_video.py --source video.mp4 --weights best.pt\n"
        )

    checkpoint_labels = [model.names[i] for i in range(checkpoint_nc)]
    if checkpoint_labels != class_names:
        print(f"Checkpoint labels: {checkpoint_labels}")
        print(f"Using --class-names order: {class_names}")
    model.model.names = check_class_names(class_names)
    return model.names, class_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO video inference")
    parser.add_argument(
        "--source",
        "-s",
        type=Path,
        required=True,
        help="Path to input video (e.g. input.mp4)",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=Path,
        default=Path(__file__).resolve().parent / "best.pt",
        help="Path to .pt weights (default: best.pt in this folder)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to save annotated video (default: <source>_detected.mp4)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device: cuda, cpu, or mps (default: auto)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show preview window while processing",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default=",".join(CLASS_NAMES),
        help="Comma-separated labels in training order (index 0, 1, ...)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.source.is_file():
        raise SystemExit(f"Video not found: {args.source}")
    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")

    out_path = args.output
    if out_path is None:
        out_path = args.source.with_name(f"{args.source.stem}_detected{args.source.suffix}")

    class_names = [n.strip() for n in args.class_names.split(",") if n.strip()]
    if not class_names:
        raise SystemExit("Provide at least one name via --class-names")

    model = YOLO(str(args.weights))
    names, class_ids = setup_model(model, class_names, args.weights)

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx = 0
    print(f"Processing: {args.source}")
    print(f"Classes: {list(names.values())}")
    print(f"Saving to: {out_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device or None,
            classes=class_ids,
            verbose=False,
        )
        annotated = results[0].plot()
        writer.write(annotated)

        if args.show:
            cv2.imshow("Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user (q).")
                break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  frames: {frame_idx}")

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Done. {frame_idx} frames -> {out_path}")


if __name__ == "__main__":
    main()
