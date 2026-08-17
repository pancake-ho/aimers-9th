from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "submission_build" / "model"
SUBMISSION_DIR = PROJECT_ROOT / "submission"
RUNTIME_PATH = PROJECT_ROOT / "src" / "runtime.py"
DIST_DIR = PROJECT_ROOT / "dist"
ZIP_PATH = DIST_DIR / "submit.zip"

MODEL_FILES = (
    "xgb_model.json",
    "cat_model.cbm",
    "bundle.pkl",
    "manifest.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_files() -> dict[str, Path]:
    files = {
        "script.py": SUBMISSION_DIR / "script.py",
        "requirements.txt": SUBMISSION_DIR / "requirements.txt",
        "model/runtime.py": RUNTIME_PATH,
    }
    files.update({f"model/{name}": MODEL_DIR / name for name in MODEL_FILES})
    return files


def _validate_artifacts(files: dict[str, Path]) -> None:
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing trained submission artifacts:\n  " + "\n  ".join(missing)
        )


def _smoke_test(zip_path: Path) -> None:
    test_path = PROJECT_ROOT / "baseline" / "data" / "test.csv"
    sample_path = PROJECT_ROOT / "baseline" / "data" / "sample_submission.csv"
    if not test_path.is_file() or not sample_path.is_file():
        print("[SMOKE] skipped: baseline/data test or sample_submission is absent")
        return

    with tempfile.TemporaryDirectory(prefix="aimers_submit_smoke_") as temp_dir:
        root = Path(temp_dir)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(root)
        (root / "data").mkdir()
        shutil.copy2(test_path, root / "data" / "test.csv")
        shutil.copy2(sample_path, root / "data" / "sample_submission.csv")
        subprocess.run([sys.executable, "script.py"], cwd=root, check=True)

        output = root / "output" / "submission.csv"
        if not output.is_file():
            raise RuntimeError("Smoke test did not produce output/submission.csv")
        print(f"[SMOKE] PASS: {output}")


def main() -> None:
    files = _required_files()
    _validate_artifacts(files)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    # The finished archive replaces an older generated archive only after all
    # trained inputs have been validated above.
    temp_zip = DIST_DIR / "submit.zip.tmp"
    if temp_zip.exists():
        temp_zip.unlink()
    with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for archive_name, source_path in files.items():
            archive.write(source_path, arcname=archive_name)

    with zipfile.ZipFile(temp_zip) as archive:
        names = archive.namelist()
        if set(names) != set(files) or len(names) != len(files):
            raise RuntimeError(f"Unexpected zip members: {names}")
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"Corrupt zip member: {corrupt}")

    temp_zip.replace(ZIP_PATH)
    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"[BUILD] {ZIP_PATH} ({size_mb:.2f} MB)")
    print(f"[BUILD] sha256={_sha256(ZIP_PATH)}")
    _smoke_test(ZIP_PATH)


if __name__ == "__main__":
    main()
