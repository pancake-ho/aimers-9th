from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "tabdpt_submission_build" / "model"
SUBMISSION_DIR = PROJECT_ROOT / "submission_tabdpt"
DIST_DIR = PROJECT_ROOT / "dist"
ZIP_PATH = DIST_DIR / "submit-tabdpt.zip"

REQUIRED_FILES = {
    "script.py": SUBMISSION_DIR / "script.py",
    "requirements.txt": SUBMISSION_DIR / "requirements.txt",
    "THIRD_PARTY_LICENSES/TabDPT-LICENSE.txt": (
        SUBMISSION_DIR / "TabDPT-LICENSE.txt"
    ),
    "THIRD_PARTY_LICENSES/NOTICE.md": SUBMISSION_DIR / "NOTICE.md",
    "model/runtime.py": PROJECT_ROOT / "src" / "runtime.py",
    "model/xgb_model.json": MODEL_DIR / "xgb_model.json",
    "model/tabdpt_context.npz": MODEL_DIR / "tabdpt_context.npz",
    "model/tabdpt1_2.safetensors": MODEL_DIR / "tabdpt1_2.safetensors",
    "model/bundle.pkl": MODEL_DIR / "bundle.pkl",
    "model/manifest.json": MODEL_DIR / "manifest.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_inputs() -> None:
    missing = [str(path) for path in REQUIRED_FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing TabDPT submission files:\n  " + "\n  ".join(missing))
    requirements = (SUBMISSION_DIR / "requirements.txt").read_text(encoding="utf-8")
    if "tabdpt==1.2.0" not in requirements or "xgboost==3.2.0" not in requirements:
        raise ValueError("TabDPT submission requirements are not exactly version-pinned.")
    if (MODEL_DIR / "tabdpt1_2.safetensors").stat().st_size < 200 * 1024 * 1024:
        raise ValueError("Bundled TabDPT checkpoint is incomplete or a Git-LFS pointer.")


def _smoke_test(zip_path: Path) -> None:
    configured = os.environ.get("AIMERS_DATA_DIR")
    data_dir = Path(configured).resolve() if configured else PROJECT_ROOT / "baseline" / "data"
    if not (data_dir / "test.csv").is_file():
        print("[SMOKE] skipped: test.csv is not available")
        return
    # A full TabDPT smoke test loads a 254 MB model and requires the optional
    # package/GPU.  Run it only when explicitly requested on the training node.
    if os.environ.get("AIMERS_TABDPT_SMOKE") != "1":
        print("[SMOKE] archive-only PASS; set AIMERS_TABDPT_SMOKE=1 for GPU inference")
        return
    with tempfile.TemporaryDirectory(prefix="aimers_tabdpt_smoke_") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(root)
        (root / "data").mkdir()
        shutil.copy2(data_dir / "test.csv", root / "data" / "test.csv")
        shutil.copy2(
            data_dir / "sample_submission.csv",
            root / "data" / "sample_submission.csv",
        )
        subprocess.run([sys.executable, "script.py"], cwd=root, check=True)
        if not (root / "output" / "submission.csv").is_file():
            raise RuntimeError("TabDPT smoke test did not create submission.csv")
        print("[SMOKE] GPU inference PASS")


def main() -> None:
    _validate_inputs()
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    temporary_zip = DIST_DIR / "submit-tabdpt.zip.tmp"
    if temporary_zip.exists():
        temporary_zip.unlink()
    with zipfile.ZipFile(
        temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for archive_name, source in REQUIRED_FILES.items():
            archive.write(source, arcname=archive_name)
    with zipfile.ZipFile(temporary_zip) as archive:
        if set(archive.namelist()) != set(REQUIRED_FILES):
            raise RuntimeError(f"Unexpected archive members: {archive.namelist()}")
        corrupt = archive.testzip()
        if corrupt:
            raise RuntimeError(f"Corrupt archive member: {corrupt}")
    temporary_zip.replace(ZIP_PATH)
    print(f"[BUILD] {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024**2:.2f} MB)")
    print(f"[BUILD] sha256={_sha256(ZIP_PATH)}")
    _smoke_test(ZIP_PATH)


if __name__ == "__main__":
    main()
