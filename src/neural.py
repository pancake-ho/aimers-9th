from __future__ import annotations

import gc
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from src.config import NeuralConfig


NEURAL_MODEL_ORDER = ("resnet", "ft_transformer")


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def resolve_device(requested: str) -> torch.device:
    name = str(requested).lower()
    if name not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Neural device must be auto, cpu or cuda; got {requested}")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Neural CUDA was requested but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch build compatible with the node driver."
        )
    return torch.device(name)


def validate_neural_backend(config: NeuralConfig) -> None:
    """Exercise every selected architecture before loading the full dataset."""
    device = resolve_device(config.device)
    selected_models = tuple(config.models)
    if not selected_models:
        raise ValueError("Neural training is enabled but no neural models were selected.")
    if len(set(selected_models)) != len(selected_models):
        raise ValueError(f"Duplicate neural models are not allowed: {selected_models}")
    unknown = [name for name in selected_models if name not in NEURAL_MODEL_ORDER]
    if unknown:
        raise ValueError(f"Unknown neural models: {unknown}")
    state = {
        "num_cols": ["n0", "n1"],
        "cat_cols": ["c0", "c1"],
        "cardinalities": [4, 5],
    }
    probe_config = replace(
        config,
        resnet_d_main=16,
        resnet_d_hidden=24,
        resnet_n_blocks=1,
        ft_d_token=8,
        ft_n_heads=2,
        ft_n_layers=1,
        ft_d_ffn=16,
    )
    x_num = torch.randn(8, 2, device=device)
    x_cat = torch.randint(0, 4, (8, 2), device=device)
    for kind in selected_models:
        spec = make_model_spec(kind, state, probe_config)
        model = build_model(spec).to(device)
        model.train()
        logits = model(x_num, x_cat)
        if logits.shape != (8,) or not torch.isfinite(logits).all():
            raise RuntimeError(f"{kind} backend preflight returned invalid output.")
        logits.square().mean().backward()
        del model, logits
    if device.type == "cuda":
        torch.cuda.synchronize()
    release_torch_memory()
    print(
        f"[BACKEND] PyTorch {torch.__version__} PASS "
        f"(device={device}, models={','.join(selected_models)})"
    )


class NeuralPreprocessor:
    """Train-only standardization layered on top of ordinal tree features.

    The shared tree preprocessor owns category vocabularies and missing-value
    imputation. This class only standardizes numerical columns and shifts the
    ordinal categories so index 0 is reserved for future/unknown categories.
    """

    def __init__(self, categorical_cols: Sequence[str], numerical_cols: Sequence[str]):
        self.cat_cols = list(categorical_cols)
        self.num_cols = list(numerical_cols)
        self.numeric_mean: np.ndarray | None = None
        self.numeric_std: np.ndarray | None = None
        self.cardinalities: list[int] = []

    def fit(self, X: pd.DataFrame) -> "NeuralPreprocessor":
        if self.num_cols:
            numerical = X[self.num_cols].to_numpy(dtype=np.float32, copy=True)
            mean = numerical.mean(axis=0, dtype=np.float64)
            std = numerical.std(axis=0, dtype=np.float64)
            std[~np.isfinite(std) | (std < 1e-6)] = 1.0
            mean[~np.isfinite(mean)] = 0.0
            self.numeric_mean = mean.astype(np.float32)
            self.numeric_std = std.astype(np.float32)
            del numerical
        else:
            self.numeric_mean = np.empty(0, dtype=np.float32)
            self.numeric_std = np.empty(0, dtype=np.float32)

        self.cardinalities = []
        for column in self.cat_cols:
            values = X[column].to_numpy(dtype=np.int64, copy=False)
            maximum = int(values.max(initial=-1))
            # Raw ordinal values are -1 (unknown) or 0..K-1 (known). After
            # adding one, the embedding table therefore needs K+1 rows.
            self.cardinalities.append(maximum + 2)
        return self

    def export_state(self) -> Dict[str, object]:
        if self.numeric_mean is None or self.numeric_std is None:
            raise RuntimeError("NeuralPreprocessor must be fitted before export.")
        return {
            "state_version": 1,
            "cat_cols": list(self.cat_cols),
            "num_cols": list(self.num_cols),
            "numeric_mean": self.numeric_mean.tolist(),
            "numeric_std": self.numeric_std.tolist(),
            "cardinalities": list(self.cardinalities),
            "numeric_clip": 8.0,
        }


def prepare_neural_arrays(
    X: pd.DataFrame,
    state: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    num_cols = list(state["num_cols"])
    cat_cols = list(state["cat_cols"])

    if num_cols:
        x_num = X[num_cols].to_numpy(dtype=np.float32, copy=True)
        mean = np.asarray(state["numeric_mean"], dtype=np.float32)
        std = np.asarray(state["numeric_std"], dtype=np.float32)
        x_num -= mean
        x_num /= std
        np.clip(x_num, -float(state.get("numeric_clip", 8.0)), float(state.get("numeric_clip", 8.0)), out=x_num)
    else:
        x_num = np.empty((len(X), 0), dtype=np.float32)

    if cat_cols:
        x_cat = X[cat_cols].to_numpy(dtype=np.int64, copy=True)
        x_cat += 1
        cardinalities = np.asarray(state["cardinalities"], dtype=np.int64)
        if (x_cat < 0).any() or (x_cat >= cardinalities[None, :]).any():
            raise ValueError("Categorical index is outside its fitted embedding range.")
    else:
        x_cat = np.empty((len(X), 0), dtype=np.int64)

    if not np.isfinite(x_num).all():
        raise ValueError("Neural numerical matrix contains NaN or infinity.")
    return np.ascontiguousarray(x_num), np.ascontiguousarray(x_cat)


class _ArrayDataset(Dataset):
    def __init__(
        self,
        x_num: np.ndarray,
        x_cat: np.ndarray,
        target: np.ndarray | None = None,
        weight: np.ndarray | None = None,
    ) -> None:
        self.x_num = torch.from_numpy(x_num)
        self.x_cat = torch.from_numpy(x_cat)
        self.target = None if target is None else torch.from_numpy(np.asarray(target, dtype=np.float32))
        self.weight = None if weight is None else torch.from_numpy(np.asarray(weight, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.x_num.shape[0])

    def __getitem__(self, index: int):
        if self.target is None:
            return self.x_num[index], self.x_cat[index]
        return self.x_num[index], self.x_cat[index], self.target[index], self.weight[index]


def _embedding_dims(cardinalities: Sequence[int]) -> list[int]:
    return [max(4, min(32, int(round(1.6 * float(cardinality) ** 0.56)))) for cardinality in cardinalities]


class _ResNetBlock(nn.Module):
    def __init__(
        self,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
    ) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(d_main)
        self.linear_first = nn.Linear(d_main, d_hidden)
        self.linear_second = nn.Linear(d_hidden, d_main)
        self.dropout_first = nn.Dropout(dropout_first)
        self.dropout_second = nn.Dropout(dropout_second)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = torch.relu(x)
        x = self.linear_first(x)
        x = torch.relu(x)
        x = self.dropout_first(x)
        x = self.linear_second(x)
        x = self.dropout_second(x)
        return residual + x


class TabularResNet(nn.Module):
    def __init__(self, spec: Mapping[str, object]) -> None:
        super().__init__()
        cardinalities = [int(value) for value in spec["cardinalities"]]
        embedding_dims = [int(value) for value in spec["embedding_dims"]]
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, dimension) for cardinality, dimension in zip(cardinalities, embedding_dims)]
        )
        input_dim = int(spec["n_num"]) + sum(embedding_dims)
        d_main = int(spec["d_main"])
        self.input = nn.Linear(input_dim, d_main)
        self.blocks = nn.ModuleList(
            [
                _ResNetBlock(
                    d_main=d_main,
                    d_hidden=int(spec["d_hidden"]),
                    dropout_first=float(spec["dropout_first"]),
                    dropout_second=float(spec["dropout_second"]),
                )
                for _ in range(int(spec["n_blocks"]))
            ]
        )
        self.head_norm = nn.BatchNorm1d(d_main)
        self.head = nn.Linear(d_main, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        pieces = [x_num]
        pieces.extend(embedding(x_cat[:, index]) for index, embedding in enumerate(self.embeddings))
        x = torch.cat(pieces, dim=1)
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        x = torch.relu(self.head_norm(x))
        return self.head(x).squeeze(1)


class FeatureTokenizer(nn.Module):
    def __init__(self, n_num: int, cardinalities: Sequence[int], d_token: int) -> None:
        super().__init__()
        self.n_num = int(n_num)
        self.n_cat = len(cardinalities)
        self.d_token = int(d_token)
        if self.n_num:
            self.num_weight = nn.Parameter(torch.empty(self.n_num, self.d_token))
            self.num_bias = nn.Parameter(torch.empty(self.n_num, self.d_token))
            nn.init.kaiming_uniform_(self.num_weight, a=math.sqrt(5))
            nn.init.uniform_(self.num_bias, -0.01, 0.01)
        else:
            self.register_parameter("num_weight", None)
            self.register_parameter("num_bias", None)

        if self.n_cat:
            offsets = np.cumsum([0, *[int(value) for value in cardinalities[:-1]]])
            self.register_buffer("cat_offsets", torch.as_tensor(offsets, dtype=torch.long))
            self.cat_embedding = nn.Embedding(sum(int(value) for value in cardinalities), self.d_token)
            self.cat_bias = nn.Parameter(torch.empty(self.n_cat, self.d_token))
            nn.init.kaiming_uniform_(self.cat_embedding.weight, a=math.sqrt(5))
            nn.init.uniform_(self.cat_bias, -0.01, 0.01)
        else:
            self.register_buffer("cat_offsets", torch.empty(0, dtype=torch.long))
            self.cat_embedding = None
            self.register_parameter("cat_bias", None)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = []
        if self.n_num:
            tokens.append(x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0))
        if self.n_cat:
            indices = x_cat + self.cat_offsets.unsqueeze(0)
            tokens.append(self.cat_embedding(indices) + self.cat_bias.unsqueeze(0))
        return torch.cat(tokens, dim=1)


class _TransformerBlock(nn.Module):
    def __init__(self, spec: Mapping[str, object]) -> None:
        super().__init__()
        d_token = int(spec["d_token"])
        self.attention_norm = nn.LayerNorm(d_token)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=int(spec["n_heads"]),
            dropout=float(spec["attention_dropout"]),
            batch_first=True,
        )
        self.attention_residual_dropout = nn.Dropout(float(spec["residual_dropout"]))
        self.ffn_norm = nn.LayerNorm(d_token)
        self.ffn_first = nn.Linear(d_token, 2 * int(spec["d_ffn"]))
        self.ffn_dropout = nn.Dropout(float(spec["ffn_dropout"]))
        self.ffn_second = nn.Linear(int(spec["d_ffn"]), d_token)
        self.ffn_residual_dropout = nn.Dropout(float(spec["residual_dropout"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(x)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        x = x + self.attention_residual_dropout(attended)
        normalized = self.ffn_norm(x)
        left, right = self.ffn_first(normalized).chunk(2, dim=-1)
        hidden = torch.relu(left) * right
        hidden = self.ffn_second(self.ffn_dropout(hidden))
        return x + self.ffn_residual_dropout(hidden)


class FTTransformer(nn.Module):
    def __init__(self, spec: Mapping[str, object]) -> None:
        super().__init__()
        d_token = int(spec["d_token"])
        self.tokenizer = FeatureTokenizer(
            n_num=int(spec["n_num"]),
            cardinalities=[int(value) for value in spec["cardinalities"]],
            d_token=d_token,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.01)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(spec) for _ in range(int(spec["n_layers"]))]
        )
        self.head_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x_num, x_cat)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([x, cls], dim=1)
        for block in self.blocks:
            x = block(x)
        x = torch.relu(self.head_norm(x[:, -1]))
        return self.head(x).squeeze(1)


def make_model_spec(
    kind: str,
    neural_state: Mapping[str, object],
    config: NeuralConfig,
) -> Dict[str, object]:
    cardinalities = [int(value) for value in neural_state["cardinalities"]]
    common: Dict[str, object] = {
        "format_version": 1,
        "kind": str(kind),
        "n_num": len(neural_state["num_cols"]),
        "cardinalities": cardinalities,
    }
    if kind == "resnet":
        common.update(
            {
                "embedding_dims": _embedding_dims(cardinalities),
                "d_main": int(config.resnet_d_main),
                "d_hidden": int(config.resnet_d_hidden),
                "n_blocks": int(config.resnet_n_blocks),
                "dropout_first": float(config.resnet_dropout_first),
                "dropout_second": float(config.resnet_dropout_second),
            }
        )
    elif kind == "ft_transformer":
        if int(config.ft_d_token) % int(config.ft_n_heads) != 0:
            raise ValueError("ft_d_token must be divisible by ft_n_heads.")
        common.update(
            {
                "d_token": int(config.ft_d_token),
                "n_heads": int(config.ft_n_heads),
                "n_layers": int(config.ft_n_layers),
                "d_ffn": int(config.ft_d_ffn),
                "attention_dropout": float(config.ft_attention_dropout),
                "ffn_dropout": float(config.ft_ffn_dropout),
                "residual_dropout": float(config.ft_residual_dropout),
            }
        )
    else:
        raise ValueError(f"Unknown neural model kind: {kind}")
    return common


def build_model(spec: Mapping[str, object]) -> nn.Module:
    kind = str(spec["kind"])
    if kind == "resnet":
        return TabularResNet(spec)
    if kind == "ft_transformer":
        return FTTransformer(spec)
    raise ValueError(f"Unknown neural model kind: {kind}")


def _batch_size(kind: str, config: NeuralConfig) -> int:
    return int(config.resnet_batch_size if kind == "resnet" else config.ft_batch_size)


def _eval_batch_size(kind: str, config: NeuralConfig) -> int:
    return int(
        config.resnet_eval_batch_size
        if kind == "resnet"
        else config.ft_eval_batch_size
    )


def _optimizer_hparams(kind: str, config: NeuralConfig) -> tuple[float, float]:
    if kind == "resnet":
        return float(config.resnet_learning_rate), float(config.resnet_weight_decay)
    if kind == "ft_transformer":
        return float(config.ft_learning_rate), float(config.ft_weight_decay)
    raise ValueError(f"Unknown neural model kind: {kind}")


def _training_limits(kind: str, config: NeuralConfig) -> tuple[int, int, int]:
    if kind == "resnet":
        return (
            int(config.resnet_max_epochs),
            int(config.resnet_min_epochs),
            int(config.resnet_early_stopping_patience),
        )
    if kind == "ft_transformer":
        return (
            int(config.ft_max_epochs),
            int(config.ft_min_epochs),
            int(config.ft_early_stopping_patience),
        )
    raise ValueError(f"Unknown neural model kind: {kind}")


def _autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    )


def _grad_scaler(device: torch.device):
    try:
        return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


@torch.inference_mode()
def predict_model(
    model: nn.Module,
    arrays: tuple[np.ndarray, np.ndarray],
    device: torch.device | str,
    batch_size: int = 8192,
    num_workers: int = 0,
) -> np.ndarray:
    device = torch.device(device)
    model.eval()
    dataset = _ArrayDataset(arrays[0], arrays[1])
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
    )
    output = []
    for x_num, x_cat in loader:
        x_num = x_num.to(device, non_blocking=True)
        x_cat = x_cat.to(device, non_blocking=True)
        with _autocast(device):
            probability = torch.sigmoid(model(x_num, x_cat))
        output.append(probability.float().cpu().numpy())
    return np.concatenate(output).astype(np.float64, copy=False)


def train_neural_fold(
    kind: str,
    train_arrays: tuple[np.ndarray, np.ndarray],
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    valid_arrays: tuple[np.ndarray, np.ndarray],
    y_valid: np.ndarray,
    neural_state: Mapping[str, object],
    config: NeuralConfig,
    random_seed: int,
) -> tuple[nn.Module, np.ndarray, int, Dict[str, object]]:
    seed_everything(random_seed)
    device = resolve_device(config.device)
    spec = make_model_spec(kind, neural_state, config)
    model = build_model(spec).to(device)
    learning_rate, weight_decay = _optimizer_hparams(kind, config)
    max_epochs, min_epochs, early_stopping_patience = _training_limits(kind, config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scaler = _grad_scaler(device)
    dataset = _ArrayDataset(train_arrays[0], train_arrays[1], y_train, sample_weight)
    loader = DataLoader(
        dataset,
        batch_size=_batch_size(kind, config),
        shuffle=True,
        drop_last=True,
        num_workers=int(config.num_workers),
        pin_memory=device.type == "cuda",
    )

    best_brier = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0
        for x_num, x_cat, target, weight in loader:
            x_num = x_num.to(device, non_blocking=True)
            x_cat = x_cat.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                logits = model(x_num, x_cat)
                per_row = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                )
                loss = (per_row * weight).sum() / weight.sum().clamp_min(1e-12)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            batch_weight = float(weight.sum().detach().cpu())
            loss_sum += float(loss.detach().cpu()) * batch_weight
            weight_sum += batch_weight

        prediction = predict_model(
            model,
            valid_arrays,
            device=device,
            batch_size=_eval_batch_size(kind, config),
            num_workers=int(config.num_workers),
        )
        brier = float(np.mean((prediction - np.asarray(y_valid, dtype=np.float64)) ** 2))
        print(
            f"[{kind}] epoch={epoch:02d} train_logloss={loss_sum / max(weight_sum, 1.0):.7f} "
            f"valid_brier={brier:.8f} pred_mean={prediction.mean():.6f}"
        )
        if brier < best_brier - 1e-7:
            best_brier = brier
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch >= min_epochs and stale_epochs >= early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError(f"{kind} did not produce a finite validation checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    prediction = predict_model(
        model,
        valid_arrays,
        device=device,
        batch_size=_eval_batch_size(kind, config),
        num_workers=int(config.num_workers),
    )
    return model, prediction, int(best_epoch), spec


def train_neural_full(
    kind: str,
    train_arrays: tuple[np.ndarray, np.ndarray],
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    neural_state: Mapping[str, object],
    config: NeuralConfig,
    random_seed: int,
    epochs: int,
) -> tuple[nn.Module, Dict[str, object]]:
    seed_everything(random_seed)
    device = resolve_device(config.device)
    spec = make_model_spec(kind, neural_state, config)
    model = build_model(spec).to(device)
    learning_rate, weight_decay = _optimizer_hparams(kind, config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scaler = _grad_scaler(device)
    dataset = _ArrayDataset(train_arrays[0], train_arrays[1], y_train, sample_weight)
    loader = DataLoader(
        dataset,
        batch_size=_batch_size(kind, config),
        shuffle=True,
        drop_last=True,
        num_workers=int(config.num_workers),
        pin_memory=device.type == "cuda",
    )
    for epoch in range(1, int(epochs) + 1):
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0
        for x_num, x_cat, target, weight in loader:
            x_num = x_num.to(device, non_blocking=True)
            x_cat = x_cat.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                logits = model(x_num, x_cat)
                per_row = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                )
                loss = (per_row * weight).sum() / weight.sum().clamp_min(1e-12)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            batch_weight = float(weight.sum().detach().cpu())
            loss_sum += float(loss.detach().cpu()) * batch_weight
            weight_sum += batch_weight
        print(f"[{kind}/final] epoch={epoch:02d}/{epochs:02d} train_logloss={loss_sum / max(weight_sum, 1.0):.7f}")
    return model, spec


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    spec: Mapping[str, object],
) -> None:
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(
        {
            "format_version": 1,
            "model_spec": dict(spec),
            "state_dict": state_dict,
        },
        str(path),
    )


def load_checkpoint(path: Path | str, device: str | torch.device = "cpu") -> nn.Module:
    try:
        checkpoint = torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(path), map_location=device)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported neural checkpoint: {checkpoint.get('format_version')}")
    model = build_model(checkpoint["model_spec"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(torch.device(device))
    model.eval()
    return model


def neural_config_state(config: NeuralConfig) -> Dict[str, object]:
    return asdict(config)


def release_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
