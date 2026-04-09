"""
package_for_colab.py — Zip the project code for upload to Google Drive / Colab.

Creates corefinst_code.zip containing everything except large data files and
generated outputs. Upload this zip to your Google Drive DRIVE_ROOT folder,
then run Step 4 of the Colab notebook.

Usage:
  python package_for_colab.py [--out corefinst_code.zip]
"""

import argparse
import os
import zipfile

# Files and directories to include
INCLUDE = [
    "src/",
    "analysis/",
    "configs/",
    "prepare_data.py",
    "train_model.py",
    "run_inference.py",
    "package_for_colab.py",
    "CorefInst_Colab.ipynb",
    "config.yaml",
    "setup.py",
    "requirements.txt",
]

# Patterns to always skip
SKIP_SUFFIXES = (".pyc", ".pyo", ".DS_Store", ".ipynb_checkpoints")
SKIP_DIRS = {"__pycache__", ".git", "processed_data", "model_output",
             "inference_output", "transmucores_data"}


def _should_skip(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    if any(p in SKIP_DIRS for p in parts):
        return True
    if any(path.endswith(s) for s in SKIP_SUFFIXES):
        return True
    return False


def create_zip(out_path: str = "corefinst_code.zip") -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    added = 0

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in INCLUDE:
            full = os.path.join(root, item)
            if os.path.isdir(full):
                for dirpath, dirnames, filenames in os.walk(full):
                    # Prune skipped dirs in-place so os.walk skips them
                    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                    for fname in filenames:
                        fpath = os.path.join(dirpath, fname)
                        rel   = os.path.relpath(fpath, root)
                        if _should_skip(rel):
                            continue
                        zf.write(fpath, rel)
                        added += 1
            elif os.path.isfile(full):
                rel = os.path.relpath(full, root)
                if not _should_skip(rel):
                    zf.write(full, rel)
                    added += 1

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Created {out_path}  ({added} files, {size_mb:.1f} MB)")
    print("\nNext steps:")
    print("  1. Upload corefinst_code.zip to your Google Drive at DRIVE_ROOT")
    print("  2. Upload transmucores_data.tar.gz to the same folder")
    print("  3. Open CorefInst_Colab.ipynb in Colab and run all cells in order")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="corefinst_code.zip")
    args = parser.parse_args()
    create_zip(args.out)
