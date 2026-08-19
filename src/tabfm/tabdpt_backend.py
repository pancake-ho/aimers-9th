from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from src.tabfm.config import TabDPTExperimentConfig


def _release_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


def validate_tabdpt_weight(path: Path | str) -> Path:
    weight_path = Path(path).expanduser().resolve()
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"TabDPT-Turbo checkpoint not found: {weight_path}. "
            "Expected the public tabdpt1_2.safetensors file."
        )
    # The public checkpoint is roughly 254 MB.  This catches Git-LFS pointer
    # files and interrupted downloads before a scheduled validation run.
    if weight_path.stat().st_size < 200 * 1024 * 1024:
        raise ValueError(
            f"TabDPT checkpoint is unexpectedly small: {weight_path.stat().st_size} bytes"
        )
    return weight_path


def validate_tabdpt_backend(weight_path: Path | str, device: str = "cuda") -> None:
    path = validate_tabdpt_weight(weight_path)
    try:
        import tabdpt
        import torch
    except ImportError as exc:
        raise ImportError(
            "tabdpt==1.2.0 and its small runtime dependencies are required."
        ) from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TabDPT experiment requires the allocated CUDA GPU.")
    version = getattr(tabdpt, "__version__", "unknown")
    print(
        f"[BACKEND] TabDPT={version} torch={torch.__version__} "
        f"device={device} weight={path}"
    )


def predict_tabdpt(
    context_x: np.ndarray,
    context_y: np.ndarray,
    query_x: np.ndarray,
    *,
    weight_path: Path | str,
    config: TabDPTExperimentConfig,
    device: str = "cuda",
) -> np.ndarray:
    """Run retrieval-free TabDPT-Turbo with one shared recent context.

    TabDPT-Turbo's attention keys/values contain context rows only; query rows
    cannot attend to one another.  Batching the official test rows therefore
    preserves the competition's row-independence requirement.
    """
    config.validate()
    path = validate_tabdpt_weight(weight_path)
    context_x = np.ascontiguousarray(context_x, dtype=np.float32)
    query_x = np.ascontiguousarray(query_x, dtype=np.float32)
    context_y = np.asarray(context_y, dtype=np.int64)
    if context_x.ndim != 2 or query_x.ndim != 2:
        raise ValueError("TabDPT context/query arrays must be two-dimensional.")
    if context_x.shape[1] != query_x.shape[1]:
        raise ValueError("TabDPT context/query feature counts do not match.")
    if context_x.shape[1] > config.max_features:
        raise ValueError("TabDPT feature count exceeds the declared maximum.")
    if len(context_x) != len(context_y):
        raise ValueError("TabDPT context feature/target lengths do not match.")
    if len(context_x) > config.context_size:
        raise ValueError("Bundled context is larger than the validated context_size.")
    if not set(np.unique(context_y)).issubset({0, 1}):
        raise ValueError("TabDPT context target must be binary.")
    if not np.isfinite(context_x).all() or not np.isfinite(query_x).all():
        raise ValueError("TabDPT input contains NaN or infinity.")

    from tabdpt import TabDPTClassifier

    model = TabDPTClassifier(
        normalizer="standard",
        missing_indicators=False,
        feature_reduction="subsample",
        context_reduction="subsample",
        device=device,
        use_flash=config.use_flash_attention,
        compile=config.compile_model,
        model_weight_path=str(path),
        verbose=False,
    )
    model.fit(context_x, context_y)
    if config.n_ensembles == 1:
        probability = model.predict_proba(
            query_x,
            context_size=None,
            batch_size=config.inference_batch_size,
            seed=config.random_seed,
        )
    else:
        probability = model.ensemble_predict_proba(
            query_x,
            n_ensembles=config.n_ensembles,
            temperature=1.0,
            context_size=None,
            batch_size=config.inference_batch_size,
            permute_classes=True,
            seed=config.random_seed,
        )
    prediction = np.asarray(probability[:, 1], dtype=np.float64)
    if prediction.shape != (len(query_x),) or not np.isfinite(prediction).all():
        raise RuntimeError("TabDPT returned invalid probabilities.")
    prediction = np.clip(prediction, 1e-6, 1.0 - 1e-6)
    del model, probability
    _release_cuda()
    return prediction
