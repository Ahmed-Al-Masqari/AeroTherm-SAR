# Forensic Dataset Integrity Audit Reports (`audit_reports/`)

This folder stores the read-only CSV/JSON audit outputs verifying zero-leakage splits and duplicate elimination across all datasets:

- **HIT-UAV v1.2.1:** Verified `2,898` normal/rotate image pairs and `105/105` official session overlap -> resolved via strict `date_captured` split.
- **UAV-COCO:** Removed `20` SHA-256 duplicate pairs with identical annotations, clipped `5` out-of-bounds boxes, and resolved `7/7` sequence overlap -> `COCO_YOLO_v2_CleanSplit`.
- **POP Dataset:** Verified `8,768` images, `26,811` person boxes, `0` MD5 duplicates, and `0` scene leakage across `19` scenes (`F1–F16`, `T1–T3`).
- **AIResQ Benchmark:** Held strictly external (`1,988` images, `2,643` person labels) for zero-shot cross-domain evaluation.

Place exported Drive audit CSV files (`audit_summary.csv`, `hit_duplicate_classification.csv`, `coco_duplicate_annotation_audit.csv`, `coco_v1_sequence_leakage.csv`) here when syncing from Google Drive.
