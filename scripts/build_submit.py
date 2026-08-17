from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = PROJECT_ROOT / "submission_build"
MODEL_DIR = BUILD_DIR / "model"
SUBMISSION_DIR = PROJECT_ROOT / "submission"
DIST_DIR = PROJECT_ROOT / "dist"
ZIP_PATH = DIST_DIR / "submit.zip"

REQUIRED_MODEL_FILES = ("lgb_model.txt", "bundle.pkl", "manifest.json")


def main():
    for name in REQUIRED_MODEL_FILES:
        path = MODEL_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Missing trained artifact: {path}")
    for name in ("script.py", "requirements.txt"):
        path = SUBMISSION_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Missing submission runtime file: {path}")

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(SUBMISSION_DIR / "script.py", arcname="script.py")
        z.write(SUBMISSION_DIR / "requirements.txt", arcname="requirements.txt")
        for name in REQUIRED_MODEL_FILES:
            z.write(MODEL_DIR / name, arcname=f"model/{name}")

    with zipfile.ZipFile(ZIP_PATH) as z:
        names = set(z.namelist())
    expected = {
        "script.py",
        "requirements.txt",
        *(f"model/{name}" for name in REQUIRED_MODEL_FILES),
    }
    if names != expected:
        raise RuntimeError(f"Unexpected zip contents: {sorted(names)}")

    size_mb = ZIP_PATH.stat().st_size / (1024 ** 2)
    print(f"[BUILD] {ZIP_PATH} ({size_mb:.2f} MB)")

    # Local 5-row smoke test using the distributed sample files.
    test_path = PROJECT_ROOT / "baseline" / "data" / "test.csv"
    sample_path = PROJECT_ROOT / "baseline" / "data" / "sample_submission.csv"
    if test_path.exists() and sample_path.exists():
        with tempfile.TemporaryDirectory(prefix="aimers_submit_smoke_") as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(ZIP_PATH) as z:
                z.extractall(root)
            (root / "data").mkdir()
            shutil.copy2(test_path, root / "data" / "test.csv")
            shutil.copy2(sample_path, root / "data" / "sample_submission.csv")
            subprocess.run([sys.executable, "script.py"], cwd=root, check=True)
            output = root / "output" / "submission.csv"
            if not output.exists():
                raise RuntimeError("Smoke test did not create output/submission.csv")
            print(f"[SMOKE] PASS: {output}")
    else:
        print("[SMOKE] skipped: local sample test/submission not found")


if __name__ == "__main__":
    main()
