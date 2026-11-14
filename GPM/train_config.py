from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union
from GPM.conditions import normalize_tech_name
from GPM.gpm import GPMTrainConfig
from GPM.paths import resolve_project_path

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "gpm_train_config.json"
_JSONC_LINE_COMMENT = re.compile(r"//.*?$", re.MULTILINE)
_JSONC_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def strip_jsonc(text: str) -> str:
    return _JSONC_BLOCK_COMMENT.sub("", _JSONC_LINE_COMMENT.sub("", text))


def default_train_config_path() -> Path:
    return _DEFAULT_CONFIG_PATH


def load_jsonc(path: Union[Path, str]) -> Any:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    return json.loads(strip_jsonc(p.read_text(encoding="utf-8")))


def load_training_settings(path: Optional[Union[Path, str]] = None) -> dict[str, Any]:
    p = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    data = load_jsonc(p)
    if not isinstance(data, dict):
        raise ValueError(f"The training configuration must be a JSON object: {p}")
    return data


def parse_loss_weights(settings: Mapping[str, Any]) -> dict[str, float]:
    lw = settings.get("loss_weights")
    if not isinstance(lw, Mapping):
        raise ValueError("gpm_train_config.json must contain a loss_weights object.")
    return {
        "expression_mse": float(lw.get("expression_mse", 1.0)),
        "ot": float(lw.get("ot", 0.05)),
    }


def load_loss_weights(path: Optional[Union[Path, str]] = None) -> dict[str, float]:
    return parse_loss_weights(load_training_settings(path))


def apply_pert_group_size(size: int) -> int:
    from Tools.Loss import set_pert_group_size
    return set_pert_group_size(size)


def build_gpm_train_config(
    *,
    config_path: Optional[Union[Path, str]] = None,
    cell_types_config_path: Optional[Union[Path, str]] = None,  # legacy, ignored
    technology_names: Sequence[str],
    overrides: Optional[Mapping[str, Any]] = None,
) -> GPMTrainConfig:
    settings = load_training_settings(config_path)
    if overrides:
        settings = {**settings, **dict(overrides)}
    epochs = max(1, int(settings["epochs"]))
    if "pert_group_size" not in settings:
        raise ValueError("gpm_train_config.json is missing pert_group_size.")
    if "min_groups" not in settings:
        raise ValueError("gpm_train_config.json is missing min_groups.")
    pert_group_size = apply_pert_group_size(int(settings["pert_group_size"]))
    if isinstance(technology_names, (str, bytes)):
        raise ValueError("technology_names must be a sequence of strings.")
    resolved_technology_names = tuple(
        dict.fromkeys(normalize_tech_name(str(name)) for name in technology_names)
    )
    if not resolved_technology_names or any(not name for name in resolved_technology_names):
        raise ValueError("At least one nonempty perturbation technology label is required.")
    return GPMTrainConfig(
        epochs=epochs,
        lr=float(settings["lr"]),
        lr_decay=float(settings.get("lr_decay", 1.0)),
        weight_decay=float(settings.get("weight_decay", 0.0)),
        device=str(settings["device"]),
        batch_size=max(1, int(settings["batch_size"])),
        pert_group_size=pert_group_size,
        seed=int(settings["seed"]),
        patience=int(settings["patience"]),
        min_groups=max(0, int(settings["min_groups"])),
        gene_embedding_dim=max(2, int(settings.get("gene_embedding_dim", 128))),
        cell_context_dim=max(1, int(settings.get("cell_context_dim", 64))),
        enhanced_node_dim=max(1, int(settings.get("enhanced_node_dim", 64))),
        edge_chunk_size=max(1, int(settings.get("edge_chunk_size", 131072))),
        technology_embedding_dim=max(1, int(settings.get("technology_embedding_dim", 32))),
        technology_names=resolved_technology_names,
        grad_clip_norm=float(settings["grad_clip_norm"]),
        loss_blur=float(settings.get("loss_blur", 0.02)),
        loss_sinkhorn_iters=max(1, int(settings.get("loss_sinkhorn_iters", 50))),
        loss_weights=parse_loss_weights(settings),
        print_every_n_groups=max(0, int(settings.get("print_every_n_groups", 20))),
        biogrid_path=resolve_project_path(settings.get("biogrid_path")),
        gene_embedding_path=resolve_project_path(settings.get("gene_embedding_path")),
        min_free_memory_gib=max(0.0, float(settings.get("min_free_memory_gib", 4.0))),
    )


def add_gpm_train_config_path_args(parser: Any) -> None:
    parser.add_argument("--train-config", type=Path, default=None, help=f"Path to the training configuration JSON (default: {default_train_config_path()}).")
    parser.add_argument("--cell-types-config", type=Path, default=None, help="Legacy option retained for compatibility. The current model ignores it.")


def add_gpm_workflow_cli_args(parser: Any) -> None:
    parser.add_argument("--force-train", action="store_true", help="Retrain the model even if a checkpoint exists.")
    parser.add_argument("--force-predict", action="store_true", help="Regenerate predictions even if saved predictions exist.")
