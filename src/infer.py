"""AeroTherm-SAR: CLI Inference Script for Thermal UAV Search & Rescue.

Supports standard fast inference (~49 FPS on T4 @ 1024px), Test-Time Augmentation
(TTA), and optional CLAHE thermal contrast enhancement (Phase 1 models).
"""

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO


def enhance_thermal_clahe(bgr_img, clip_limit: float = 2.5, tile_size=(8, 8)):
    """Apply CLAHE local contrast enhancement on a thermal image."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced_gray = clahe.apply(gray)
    return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)


def parse_args():
    parser = argparse.ArgumentParser(description="AeroTherm-SAR Thermal Inference")
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLO11 .pt weights")
    parser.add_argument("--source", type=str, required=True, help="Image, directory, or video path")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference resolution (1024 for Phase 2, 800/640 for Phase 1)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (Multi-Scale)")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE preprocessing (recommended for Phase 1 HIT-UAV models)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ID or 'cpu'")
    parser.add_argument("--project", type=str, default="runs/predict", help="Output directory")
    parser.add_argument("--name", type=str, default="sar_infer", help="Run name")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    source = args.source

    if args.clahe and Path(source).is_file() and Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        img = cv2.imread(source)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        source = enhance_thermal_clahe(img)

    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        augment=args.tta,
        device=args.device,
        save=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    total_dets = sum(len(r.boxes) for r in results if r.boxes is not None)
    print(f"Inference complete. Processed {len(results)} frame(s) | Total detections: {total_dets}")


if __name__ == "__main__":
    main()
