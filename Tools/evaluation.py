"""Evaluate control-referenced perturbation effects over the full panel and per-condition top genes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union
import numpy as np
import scipy.sparse as sps
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.metrics import mean_squared_error, r2_score

from GPM.conditions import is_control_condition


METRIC_NAMES: tuple[str, ...] = (
    "MSE",
    "MSE.Norm",
    "PCC",
    "R2",
    "Spearman",
    "Wasserstein",
)

# Rank response genes separately for each complete perturbation condition.
REPORT_SECTIONS: tuple[tuple[str, str], ...] = (
    ("all", "All genes in the expression panel"),
    ("top20", "Top 20 genes by absolute observed perturbation effect"),
    ("top50", "Top 50 genes by absolute observed perturbation effect"),
)


@dataclass
class MetricBundle:
    """Metrics for perturbation effect predictions."""

    MSE: float = float("nan")
    MSE_Norm: float = field(default=float("nan"), metadata={"alias": "MSE.Norm"})
    PCC: float = float("nan")
    R2: float = float("nan")
    Spearman: float = float("nan")
    Wasserstein: float = float("nan")
    n_perts: int = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "MSE": self.MSE,
            "MSE.Norm": self.MSE_Norm,
            "PCC": self.PCC,
            "R2": self.R2,
            "Spearman": self.Spearman,
            "Wasserstein": self.Wasserstein,
            "n_perts": float(self.n_perts),
        }


@dataclass
class EvaluationReport:
    """Evaluation results for the full panel and per-condition Top 20 and Top 50 genes."""

    dataset_name: str
    sections: dict[str, MetricBundle] = field(default_factory=dict)
    gene_panel_size: int = 0

    @property
    def metrics(self) -> MetricBundle:
        return self.sections.get("all", MetricBundle())

    def get(self, section: str) -> MetricBundle:
        return self.sections.get(section, MetricBundle())


def _aggregate_cells(arr: np.ndarray) -> np.ndarray:
    """Average cell-level values and preserve already aggregated vectors."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        return np.mean(a, axis=0)
    raise ValueError(f"Expected a one- or two-dimensional array, got shape {a.shape}.")


def canonical_gpm_perturbation_key(pert: str) -> str:
    """Normalize a perturbation label to the GPM condition format."""
    from Data.data_tools import build_condition_label

    cond, _ = build_condition_label(str(pert).strip())
    return cond


def _rekey_pert_effect_dict(
    mapping: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Reindex effects by canonical perturbation labels, merging equivalent keys."""
    buckets: dict[str, list[np.ndarray]] = {}
    for key, vec in mapping.items():
        ck = canonical_gpm_perturbation_key(key)
        buckets.setdefault(ck, []).append(np.asarray(vec, dtype=np.float32))
    return {
        k: np.mean(vs, axis=0).astype(np.float32) if len(vs) > 1 else vs[0].astype(np.float32)
        for k, vs in buckets.items()
    }


def aggregate_pred_expression_to_pert_effect(
    perturbations: Sequence[str],
    pred_expressions: np.ndarray,
    ctrl_mean: np.ndarray,
) -> dict[str, np.ndarray]:
    """Average predictions by condition and subtract the control mean."""
    ctrl = np.asarray(ctrl_mean, dtype=np.float64).ravel()
    pred_out: dict[str, list[np.ndarray]] = {}
    for pert, expr in zip(perturbations, pred_expressions):
        pred_out.setdefault(str(pert), []).append(np.asarray(expr, dtype=np.float64))
    return {
        k: (np.mean(v, axis=0) - ctrl).astype(np.float32)
        for k, v in pred_out.items()
    }


def aggregate_predictions_by_perturbation(
    perturbations: Sequence[str],
    pred_deltas: np.ndarray,
    true_deltas: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Average the cell-level values within each perturbation."""
    pred_out: dict[str, list[np.ndarray]] = {}
    true_out: dict[str, list[np.ndarray]] = {}
    for pert, p, t in zip(perturbations, pred_deltas, true_deltas):
        key = str(pert)
        pred_out.setdefault(key, []).append(np.asarray(p, dtype=np.float64))
        true_out.setdefault(key, []).append(np.asarray(t, dtype=np.float64))
    return (
        {k: np.mean(v, axis=0).astype(np.float32) for k, v in pred_out.items()},
        {k: np.mean(v, axis=0).astype(np.float32) for k, v in true_out.items()},
    )


def align_pred_and_truth(
    pred_by_pert: Mapping[str, np.ndarray],
    true_by_pert: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Align predictions and observations by shared canonical perturbation labels."""
    pred_norm = _rekey_pert_effect_dict(pred_by_pert)
    true_norm = _rekey_pert_effect_dict(true_by_pert)
    common = sorted(set(pred_norm) & set(true_norm))
    if not common:
        raise ValueError("Predicted and observed perturbation labels have no overlap.")
    n_drop_pred = len(set(pred_norm) - set(true_norm))
    n_drop_true = len(set(true_norm) - set(pred_norm))
    if n_drop_pred or n_drop_true:
        print(
            f"[evaluation] Matched {len(common)} perturbations "
            f"(pred-only={n_drop_pred}, truth-only={n_drop_true})"
        )
    return (
        {k: np.asarray(pred_norm[k], dtype=np.float32) for k in common},
        {k: np.asarray(true_norm[k], dtype=np.float32) for k in common},
    )


def _mean_expression_matrix(adata: Any, row_mask: np.ndarray) -> np.ndarray:
    x = adata[row_mask].X
    if sps.issparse(x):
        return np.asarray(x.mean(axis=0)).ravel().astype(np.float64)
    return np.asarray(x, dtype=np.float64).mean(axis=0).ravel()


def build_true_delta_by_gpm_condition(
    adata: Any,
    ctrl_mean: np.ndarray,
    panel_gene_names: Sequence[str],
    *,
    split: str = "test",
    condition_col: str = "gpm_condition",
) -> dict[str, np.ndarray]:
    """Compute observed test effects by subtracting the control mean from each condition mean."""
    split_col = "gpm_split" if "gpm_split" in adata.obs.columns else "split"
    if split_col not in adata.obs.columns:
        raise ValueError(f"AnnData is missing the {split_col!r} observation column.")
    if condition_col not in adata.obs.columns:
        raise ValueError(f"AnnData is missing the {condition_col!r} observation column.")

    var_genes = (
        adata.var["gene_name"].astype(str).tolist()
        if "gene_name" in adata.var.columns
        else adata.var_names.astype(str).tolist()
    )
    name_to_var = {g: i for i, g in enumerate(var_genes)}
    ctrl_vec = np.asarray(ctrl_mean, dtype=np.float32).ravel()
    if ctrl_vec.shape[0] != len(panel_gene_names):
        raise ValueError(
            f"Control mean has {ctrl_vec.shape[0]} entries, but the gene panel contains {len(panel_gene_names)} genes."
        )
    var_idx = [int(name_to_var[g]) for g in panel_gene_names]

    mask = adata.obs[split_col].astype(str) == str(split)
    sub = adata[mask]
    out: dict[str, np.ndarray] = {}
    for cond in sorted(sub.obs[condition_col].astype(str).unique(), key=str):
        if is_control_condition(cond):
            continue
        row_mask = (sub.obs[condition_col].astype(str) == cond).to_numpy()
        if not row_mask.any():
            continue
        expr = _mean_expression_matrix(sub, row_mask)[var_idx]
        out[str(cond)] = (expr - ctrl_vec).astype(np.float32)
    return out


def compute_no_change_mse_baseline(y_true_list: Sequence[np.ndarray]) -> float:
    """Return the MSE of predicting zero change for the observed effects."""
    if not y_true_list:
        return float("nan")
    all_t = np.concatenate(
        [np.asarray(t, dtype=np.float64).ravel() for t in y_true_list]
    )
    return float(np.mean(all_t**2))


def compute_metrics_on_vectors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    mse_baseline: Optional[float] = None,
) -> dict[str, float]:
    """Compute metrics for a pair of gene-wise effect vectors."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"Observed and predicted vectors have different shapes: {yt.shape} and {yp.shape}.")

    mse_val = float(mean_squared_error(yt, yp))
    b = float(mse_baseline) if mse_baseline is not None else float(np.mean(yt**2))
    mse_norm = mse_val / b if b > 1e-12 else float("nan")

    if np.std(yt) < 1e-12 or np.std(yp) < 1e-12:
        pcc_val = float("nan")
        spear_val = float("nan")
    else:
        pcc_val = float(np.corrcoef(yt, yp)[0, 1])
        spear_val = float(spearmanr(yt, yp).correlation)

    try:
        r2_val = float(r2_score(yt, yp))
    except ValueError:
        r2_val = float("nan")

    try:
        wdist = float(wasserstein_distance(yt, yp))
    except Exception:
        wdist = float("nan")

    return {
        "MSE": mse_val,
        "MSE.Norm": mse_norm,
        "PCC": pcc_val,
        "R2": r2_val,
        "Spearman": spear_val,
        "Wasserstein": wdist,
    }


def _metrics_mean_over_perts(per_pert_metrics: list[dict[str, float]]) -> MetricBundle:
    if not per_pert_metrics:
        return MetricBundle()

    out: dict[str, list[float]] = {k: [] for k in METRIC_NAMES}
    for m in per_pert_metrics:
        for k in METRIC_NAMES:
            v = m.get(k, float("nan"))
            if v is not None and np.isfinite(v):
                out[k].append(float(v))

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return MetricBundle(
        MSE=_mean(out["MSE"]),
        MSE_Norm=_mean(out["MSE.Norm"]),
        PCC=_mean(out["PCC"]),
        R2=_mean(out["R2"]),
        Spearman=_mean(out["Spearman"]),
        Wasserstein=_mean(out["Wasserstein"]),
        n_perts=len(per_pert_metrics),
    )


def _bundle_from_pert_vectors(
    y_true_list: list[np.ndarray],
    y_pred_list: list[np.ndarray],
) -> MetricBundle:
    if not y_true_list:
        return MetricBundle()

    baseline = compute_no_change_mse_baseline(y_true_list)
    per_pert = [
        compute_metrics_on_vectors(t, p, mse_baseline=baseline)
        for t, p in zip(y_true_list, y_pred_list)
    ]
    bundle = _metrics_mean_over_perts(per_pert)

    all_t = np.concatenate([np.asarray(t, dtype=np.float64).ravel() for t in y_true_list])
    all_p = np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in y_pred_list])
    sse = float(np.sum((all_t - all_p) ** 2))
    sum_true_sq = float(np.sum(all_t**2))
    bundle.MSE_Norm = (
        sse / sum_true_sq if sum_true_sq > 1e-12 else float("nan")
    )
    return bundle


def _print_metrics(bundle: MetricBundle) -> None:
    """Print the metric values."""
    d = bundle.as_dict()

    def _fmt(name: str) -> str:
        v = d[name]
        if not np.isfinite(v):
            return "nan"
        return f"{float(v):.6f}"

    for name in METRIC_NAMES:
        print(f"  {name}: {_fmt(name)}")


def evaluate_perturbation_effects(
    dataset_name: str,
    predictions: Mapping[str, np.ndarray],
    truths: Mapping[str, np.ndarray],
    gene_names: Sequence[str],
) -> EvaluationReport:
    """Evaluate control-referenced effects over the full panel and per-condition top genes.

    Rank genes by absolute observed effect within each complete condition. Skip
    controls.
    """
    if set(predictions.keys()) != set(truths.keys()):
        raise ValueError("Predicted and observed perturbation labels must match.")

    vectors: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        key: ([], []) for key, _ in REPORT_SECTIONS
    }

    for pert in sorted(predictions.keys()):
        if is_control_condition(pert):
            continue

        y_true = _aggregate_cells(truths[pert])
        y_pred = _aggregate_cells(predictions[pert])
        if y_true.shape[0] != len(gene_names):
            raise ValueError(
                f"Perturbation {pert!r} has {y_true.shape[0]} effect values for a panel of {len(gene_names)} genes."
            )
        vectors["all"][0].append(y_true)
        vectors["all"][1].append(y_pred)
        ranking = np.argsort(-np.abs(y_true), kind="stable")
        for size, section in ((20, "top20"), (50, "top50")):
            selected = ranking[: min(size, ranking.size)]
            vectors[section][0].append(y_true[selected])
            vectors[section][1].append(y_pred[selected])

    return EvaluationReport(
        dataset_name=dataset_name,
        gene_panel_size=len(gene_names),
        sections={
            key: _bundle_from_pert_vectors(*vectors[key])
            for key, _ in REPORT_SECTIONS
        },
    )


def print_evaluation_report(
    report: EvaluationReport,
    *,
    dataset_name: Optional[str] = None,
) -> None:
    """Print metrics for each gene subset."""
    ds = dataset_name or report.dataset_name
    print(f"\n========== {ds} | Genes: {report.gene_panel_size} | Perturbations: {report.metrics.n_perts} ==========")
    for section, label in REPORT_SECTIONS:
        print(f"[{section}] {label}")
        _print_metrics(report.get(section))

def load_test_predictions_npz(path: Union[Path, str]) -> dict[str, Any]:
    """Load saved test predictions."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Test prediction archive not found: {p}")
    with np.load(p, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def evaluate_from_test_npz(
    npz_path: Union[Path, str],
    adata: Any,
    ctrl_mean: np.ndarray,
    *,
    dataset_key: str,
    gene_names: Optional[Sequence[str]] = None,
    print_report: bool = True,
) -> tuple[EvaluationReport, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Evaluate saved test predictions against effects reconstructed from observed cells.

    Use the saved ``pred_delta`` when available, otherwise subtract the control mean
    from ``pred_expression``.
    """
    raw = load_test_predictions_npz(npz_path)
    gene_names_npz = [str(g) for g in raw["gene_names"].tolist()]
    if gene_names is not None:
        gene_names_use = [str(g) for g in gene_names]
        if gene_names_use != gene_names_npz:
            raise ValueError(
                f"The NPZ gene panel does not match gene_names (NPZ: {len(gene_names_npz)}, requested: {len(gene_names_use)})."
            )
    else:
        gene_names_use = gene_names_npz

    pert_arr = raw["perturbation"].astype(str)
    if "pred_delta" in raw:
        pred_delta = np.asarray(raw["pred_delta"], dtype=np.float32)
        if pred_delta.shape[0] != pert_arr.shape[0]:
            raise ValueError("The number of pred_delta rows must match the number of perturbation labels in the NPZ archive.")
        true_delta_npz = np.asarray(
            raw["true_delta"] if "true_delta" in raw else pred_delta,
            dtype=np.float32,
        )
        pred_by_pert, _ = aggregate_predictions_by_perturbation(
            pert_arr.tolist(),
            pred_delta,
            true_delta_npz,
        )
        pred_src = "pred_delta"
    elif "pred_expression" in raw:
        pred_expression = np.asarray(raw["pred_expression"], dtype=np.float32)
        if pred_expression.shape[0] != pert_arr.shape[0]:
            raise ValueError("The number of pred_expression rows must match the number of perturbation labels in the NPZ archive.")
        pred_by_pert = aggregate_pred_expression_to_pert_effect(
            pert_arr.tolist(),
            pred_expression,
            ctrl_mean,
        )
        pred_src = "pred_expression−ctrl_mean"
    else:
        raise KeyError("The NPZ archive must contain pred_delta or pred_expression.")

    true_by_pert = build_true_delta_by_gpm_condition(
        adata, ctrl_mean, gene_names_use, split="test"
    )
    pred_by_pert, true_by_pert = align_pred_and_truth(pred_by_pert, true_by_pert)
    print(
        f"[evaluation] Observed effects: test condition mean expression minus the control mean (ctrl_mean.npz) | Predictions: {pred_src}"
    )

    if print_report:
        report = evaluate_and_print(
            dataset_key,
            pred_by_pert,
            true_by_pert,
            gene_names_use,
        )
    else:
        report = evaluate_perturbation_effects(
            dataset_key,
            pred_by_pert,
            true_by_pert,
            gene_names_use,
        )
    return report, pred_by_pert, true_by_pert


def evaluate_and_print(
    dataset_name: str,
    predictions: Mapping[str, np.ndarray],
    truths: Mapping[str, np.ndarray],
    gene_names: Sequence[str],
) -> EvaluationReport:
    """Evaluate perturbation effects and print the metric report."""
    report = evaluate_perturbation_effects(
        dataset_name,
        predictions,
        truths,
        gene_names,
    )
    print_evaluation_report(report)
    return report


def evaluation_report_to_dict(report: EvaluationReport) -> dict[str, Any]:
    sections: dict[str, Any] = {}
    for key, bundle in report.sections.items():
        sections[key] = bundle.as_dict()
    return {
        "dataset_name": report.dataset_name,
        "gene_panel_size": report.gene_panel_size,
        "sections": sections,
    }


def average_evaluation_reports(
    reports: Sequence[EvaluationReport],
    *,
    label: str = "Mean across datasets",
) -> EvaluationReport:
    if not reports:
        raise ValueError("At least one evaluation report is required.")

    avg = EvaluationReport(
        dataset_name=label,
        gene_panel_size=reports[0].gene_panel_size,
    )

    def _nanmean(vals: list[float]) -> float:
        xs = [float(v) for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(xs)) if xs else float("nan")

    for section_key, _ in REPORT_SECTIONS:
        bundles = [r.get(section_key) for r in reports if r.get(section_key).n_perts > 0]
        if not bundles:
            avg.sections[section_key] = MetricBundle()
            continue
        avg.sections[section_key] = MetricBundle(
            MSE=_nanmean([b.MSE for b in bundles]),
            MSE_Norm=_nanmean([b.MSE_Norm for b in bundles]),
            PCC=_nanmean([b.PCC for b in bundles]),
            R2=_nanmean([b.R2 for b in bundles]),
            Spearman=_nanmean([b.Spearman for b in bundles]),
            Wasserstein=_nanmean([b.Wasserstein for b in bundles]),
            n_perts=int(round(np.mean([b.n_perts for b in bundles]))),
        )
    return avg


def cross_cell_metrics_payload(
    *,
    seed: int,
    stem: str,
    bundle: Mapping[str, Any],
    reports: Mapping[str, EvaluationReport],
    average_report: EvaluationReport,
    config_fields: Mapping[str, Any],
    npz_paths: Mapping[str, str],
    checkpoint: str,
    best_checkpoint: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "intersection_across_cell",
        "seed": seed,
        "pairing_seed": seed,
        "artifact_stem": stem,
        "dataset_keys": list(bundle.get("dataset_keys") or []),
        "n_intersection_genes": bundle.get("n_intersection_genes"),
        "excluded_dataset": bundle.get("excluded_dataset"),
        "data_source": bundle.get("data_source"),
        "kd_top_genes_pkl": bundle.get("_kd_top_genes_pkl") or bundle.get("kd_top_genes_pkl"),
        "per_dataset": {key: evaluation_report_to_dict(rep) for key, rep in reports.items()},
        "average_reference": evaluation_report_to_dict(average_report),
        "test_npz_by_dataset": dict(npz_paths),
        "checkpoint": checkpoint,
        "best_checkpoint": best_checkpoint,
    }
    payload.update(dict(config_fields))
    return payload


def cross_tech_metrics_payload(
    *,
    seed: int,
    stem: str,
    bundle: Mapping[str, Any],
    reports: Mapping[str, EvaluationReport],
    config_fields: Mapping[str, Any],
    npz_paths: Mapping[str, str],
    checkpoint: str,
    best_checkpoint: str,
) -> dict[str, Any]:
    return {
        "experiment": "mixed_technologies_unseen_perturbations",
        "seed": seed,
        "pairing_seed": seed,
        "artifact_stem": stem,
        "dataset_keys": list(bundle.get("dataset_keys") or []),
        "perturbation_teches": list(bundle.get("perturbation_teches") or []),
        "n_panel_genes": bundle.get("n_panel_genes"),
        "n_raw_symbol_intersection": bundle.get("n_raw_symbol_intersection"),
        "exp4_export_dir": bundle.get("exp4_export_dir"),
        "exp4_genes_npz": bundle.get("exp4_genes_npz"),
        "per_dataset": {key: evaluation_report_to_dict(rep) for key, rep in reports.items()},
        "test_npz_by_dataset": dict(npz_paths),
        "checkpoint": checkpoint,
        "best_checkpoint": best_checkpoint,
        **dict(config_fields),
    }


def unseen_cell_metrics_payload(
    *,
    seed: int,
    stem: str,
    bundle: Mapping[str, Any],
    report: EvaluationReport,
    config_fields: Mapping[str, Any],
    npz_path: str,
    checkpoint: str,
    best_checkpoint: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "intersection_unseen_cell",
        "seed": seed,
        "pairing_seed": seed,
        "artifact_stem": stem,
        "fold_label": bundle.get("fold_label"),
        "fold_slug": bundle.get("fold_slug"),
        "held_out_dataset": bundle.get("held_out_dataset"),
        "held_out_cell_line": bundle.get("held_out_cell_line"),
        "train_dataset_keys": list(bundle.get("train_dataset_keys") or []),
        "dataset_keys": list(bundle.get("dataset_keys") or []),
        "n_intersection_genes": bundle.get("n_intersection_genes"),
        "data_source": bundle.get("data_source"),
        "kd_top_genes_pkl": bundle.get("_kd_top_genes_pkl") or bundle.get("kd_top_genes_pkl"),
        "test_report": evaluation_report_to_dict(report),
        "test_npz": npz_path,
        "checkpoint": checkpoint,
        "best_checkpoint": best_checkpoint,
    }
    payload.update(dict(config_fields))
    return payload
