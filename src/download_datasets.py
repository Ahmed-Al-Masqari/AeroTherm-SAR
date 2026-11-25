"""AeroTherm-SAR: Resume-Safe Zenodo Dataset Downloader with MD5 Verification.

Downloads the 4 verified Zenodo open-access thermal UAV datasets used in Phase 2
without modifying raw archives.
"""

import argparse
import hashlib
from pathlib import Path
import requests
from tqdm import tqdm


ZENODO_RECORDS = {
    "HIT-UAV": "7633134",
    "UAV_Thermal_Human_Detection": "21515906",
    "AIResQ_Benchmark": "17405074",
    "UAV_Thermal_Videos": "4327118",
}


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_all(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    for dataset_name, record_id in ZENODO_RECORDS.items():
        print(f"\n=== {dataset_name} (Zenodo Record {record_id}) ===")
        resp = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
        resp.raise_for_status()
        record = resp.json()

        ds_dir = output_root / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        for item in record.get("files", []):
            filename = Path(item["key"]).name
            dest = ds_dir / filename
            expected_size = item.get("size", 0)
            checksum = item.get("checksum", "")

            if dest.exists() and dest.stat().st_size == expected_size:
                if checksum.startswith("md5:") and md5_file(dest) == checksum.split(":", 1)[1]:
                    print(f"[SKIP] Verified existing file: {filename}")
                    continue

            url = item["links"]["self"]
            with requests.get(url, stream=True, timeout=(30, 180)) as r:
                r.raise_for_status()
                with dest.open("wb") as f, tqdm(
                    total=expected_size, unit="B", unit_scale=True, desc=filename
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            if checksum.startswith("md5:"):
                actual = md5_file(dest)
                expected = checksum.split(":", 1)[1]
                if actual != expected:
                    raise ValueError(f"MD5 mismatch for {filename}: {actual} != {expected}")
                print(f"[OK] MD5 verified: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Download AeroTherm-SAR Zenodo datasets")
    parser.add_argument("--out", type=str, default="raw_datasets", help="Target directory")
    args = parser.parse_args()
    download_all(Path(args.out))


if __name__ == "__main__":
    main()
