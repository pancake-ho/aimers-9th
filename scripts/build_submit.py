from __future__ import annotations

import hashlib
import json
import os
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
NEURAL_RUNTIME_PATH = PROJECT_ROOT / "src" / "neural.py"
DIST_DIR = PROJECT_ROOT / "dist"
ZIP_PATH = DIST_DIR / "submit.zip"

CORE_MODEL_FILES = (
    "xgb_model.json",
    "lgb_model.txt",
    "cat_model.cbm",
    "bundle.pkl",
    "manifest.json",
)

# DACON preinstalls these runtime libraries. Listing a different version in
# submission/requirements.txt can replace the CUDA/Python-matched base package
# and cause an installation failure before script.py starts.
EVALUATION_PROVIDED_REQUIREMENTS = {
    "joblib",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "threadpoolctl",
    "torch",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_files() -> dict[str, Path]:
    files = {
        "script.py": (
            SUBMISSION_DIR / "script.py"
        ),
        "requirements.txt": (
            SUBMISSION_DIR
            / "requirements.txt"
        ),
        "model/runtime.py": (
            RUNTIME_PATH
        ),
    }

    files.update(
        {
            f"model/{name}": (
                MODEL_DIR / name
            )
            for name in CORE_MODEL_FILES
        }
    )

    manifest_path = (
        MODEL_DIR / "manifest.json"
    )

    # During very early preflight the trained
    # manifest may not exist yet.
    if not manifest_path.is_file():
        return files

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)

    model_order = tuple(
        manifest.get(
            "model_order",
            (),
        )
    )

    supported_orders = {
        (
            "xgb",
            "lgb",
            "cat",
        ),
        (
            "xgb",
            "lgb",
            "cat",
            "resnet",
        ),
        (
            "xgb",
            "lgb",
            "cat",
            "resnet",
            "ft_transformer",
        ),
    }

    if model_order not in supported_orders:
        raise ValueError(
            "Unsupported manifest "
            f"model_order: {model_order}"
        )

    # Do not derive neural models using
    # a hard-coded GBDT slice.
    neural_names = [
        name
        for name in model_order
        if name
        in {
            "resnet",
            "ft_transformer",
        }
    ]

    if neural_names:
        files[
            "model/neural_runtime.py"
        ] = NEURAL_RUNTIME_PATH

        for name in neural_names:
            files[
                f"model/{name}.pt"
            ] = (
                MODEL_DIR
                / f"{name}.pt"
            )

    return files


def _validate_artifacts(files: dict[str, Path]) -> None:
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing trained submission artifacts:\n  " + "\n  ".join(missing)
        )
    _validate_submission_requirements(files["requirements.txt"])


def _validate_submission_requirements(path: Path) -> None:
    forbidden = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        normalized = line.lower().replace("_", "-")
        name = normalized
        for marker in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
            name = name.split(marker, 1)[0].strip()
        if name in EVALUATION_PROVIDED_REQUIREMENTS:
            forbidden.append(line)
    if forbidden:
        raise ValueError(
            "submission/requirements.txt must use DACON's preinstalled runtime "
            f"packages instead of reinstalling them: {forbidden}"
        )


def _smoke_test(zip_path: Path) -> None:
    configured_data_dir = os.environ.get("AIMERS_DATA_DIR")
    data_dir = (
        Path(configured_data_dir).expanduser().resolve()
        if configured_data_dir
        else PROJECT_ROOT / "baseline" / "data"
    )
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
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
