"""AeroTherm-SAR: BoT-SORT + Sparse Optical Flow GMC Tactical Tracker.

Compensates for UAV camera ego-motion using Lucas-Kanade optical flow,
classifies tracked survivors as MOVING vs. STATIC, and renders a non-occluding
top-padded tactical telemetry HUD.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


PALETTE = {
    0: (40, 60, 255),   # Person (Red)
    1: (255, 160, 40),  # Car (Cyan-Blue)
    2: (50, 220, 100),  # Bicycle (Green)
    3: (40, 200, 255),  # OtherVehicle (Gold)
}


def estimate_camera_shift(prev_gray, curr_gray):
    """Estimate median (dx, dy) camera ego-motion via Lucas-Kanade Optical Flow."""
    if prev_gray is None:
        return 0.0, 0.0
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=120, qualityLevel=0.01, minDistance=20)
    if p0 is None:
        return 0.0, 0.0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
    if p1 is None or st is None:
        return 0.0, 0.0
    good_old = p0[st == 1]
    good_new = p1[st == 1]
    if len(good_old) < 5:
        return 0.0, 0.0
    diffs = good_new - good_old
    return float(np.median(diffs[:, 0])), float(np.median(diffs[:, 1]))


def draw_tactical_hud(frame_bgr, res, prev_gray, pos_history, motion_thresh=2.5):
    curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cam_dx, cam_dy = estimate_camera_shift(prev_gray, curr_gray)

    pad_top = 60
    canvas = cv2.copyMakeBorder(frame_bgr, pad_top, 0, 0, 0, cv2.BORDER_CONSTANT, value=(15, 20, 28))

    total_objects = 0
    moving_survivors = 0
    static_survivors = 0

    if res.boxes is not None and len(res.boxes) > 0:
        boxes = res.boxes.xyxy.cpu().tolist()
        clss = res.boxes.cls.cpu().tolist()
        ids = res.boxes.id.cpu().tolist() if res.boxes.id is not None else [None] * len(boxes)

        for box, cls, tid in zip(boxes, clss, ids):
            cid = int(cls)
            x1, y1, x2, y2 = [int(v) for v in box]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            total_objects += 1

            status = "STATIC"
            if tid is not None:
                tid = int(tid)
                if pos_history[tid]:
                    px, py = pos_history[tid][-1]
                    rel_disp = np.hypot((cx - px) - cam_dx, (cy - py) - cam_dy)
                    if rel_disp > motion_thresh:
                        status = "MOVING"
                pos_history[tid].append((cx, cy))
                pos_history[tid] = pos_history[tid][-15:]

            if cid == 0:
                if status == "MOVING":
                    moving_survivors += 1
                else:
                    static_survivors += 1

            color = PALETTE.get(cid, (255, 255, 255))
            cv2.rectangle(canvas, (x1, y1 + pad_top), (x2, y2 + pad_top), color, 2 if cid == 0 else 1)
            tag = f"ID:{tid} {status}" if tid is not None else status
            if total_objects <= 14 or cid == 0:
                cv2.putText(canvas, tag, (x1, max(pad_top + 12, y1 + pad_top - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    if total_objects == 0:
        scene_lbl, hdr_col = "SCENE: EMPTY TERRAIN (CLEAR)", (180, 220, 180)
    elif total_objects <= 8:
        scene_lbl, hdr_col = f"SCENE: SEMI-CROWDED ({total_objects} Targets)", (80, 210, 255)
    else:
        scene_lbl, hdr_col = f"SCENE: HIGH DENSITY CROWD ({total_objects} Targets)", (80, 90, 255)

    cv2.putText(canvas, f"{scene_lbl} | DRONE SHIFT: dx={cam_dx:.1f}, dy={cam_dy:.1f}px",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, hdr_col, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"SURVIVORS -> Moving: {moving_survivors} | Static/Trapped: {static_survivors} | Total Active: {moving_survivors + static_survivors}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 235, 240), 1, cv2.LINE_AA)
    return canvas, curr_gray


def main():
    parser = argparse.ArgumentParser(description="AeroTherm-SAR BoT-SORT GMC Video Tracker")
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLO11 .pt weights")
    parser.add_argument("--source", type=str, required=True, help="Input video file path")
    parser.add_argument("--tracker", type=str, default="configs/sar_botsort_gmc.yaml", help="Tracker YAML path")
    parser.add_argument("--output", type=str, default="outputs/sar_tracked_hud.mp4", help="Output MP4 path")
    parser.add_argument("--imgsz", type=int, default=800, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="0", help="CUDA device or 'cpu'")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h + 60))

    model = YOLO(args.weights)
    prev_gray = None
    pos_history = defaultdict(list)

    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.45,
            agnostic_nms=True,
            tracker=args.tracker,
            persist=True,
            device=args.device,
            verbose=False,
        )[0]
        hud_frame, prev_gray = draw_tactical_hud(frame, res, prev_gray, pos_history)
        writer.write(hud_frame)
        frames += 1

    cap.release()
    writer.release()
    print(f"Tracking complete ({frames} frames). Saved tactical HUD video to: {out_path}")


if __name__ == "__main__":
    main()
