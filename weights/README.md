# Trained Model Weights (`weights/`)

To keep this Git repository lightweight and fast to clone, binary `.pt` checkpoints are excluded from Git tracking (`.gitignore`) and published via **GitHub Releases**:

- **Release `v2.0-cross-domain-champion`** -> `exp02_ep35_champion_1024.pt` (`19.2 MB`, `YOLO11s @ 1024px`, `52.82%` Std / `54.96%` TTA on External `AIResQ`)
- **Release `v1.0-tactical-hit-uav`** -> `yolo11s_pro_800px_best.pt` (`18.30 MB`), `yolo11s_640px_best.pt` (`18.28 MB`), `best_thermal_unet.pt` (`7.39 MB`)

Download the desired `.pt` file from the repository's **Releases** page and place it inside this `weights/` directory before running `src/infer.py` or `src/track_botsort.py`.
