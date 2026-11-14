"""Single source of truth for the retained perturbation datasets."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Optional, Sequence, Union
import re
import unicodedata
import anndata as ad

DATASET_MODULES = {
    "replogle_rpe1": "Data.replogle_rpe1",
    "replogle_k562": "Data.replogle_k562",
    "nadig_hepg2": "Data.nadig_hepg2",
    "nadig_jurkat": "Data.nadig_jurkat",
    "tian_crispri": "Data.tian_crispri",
    "xu": "Data.Xu",
    "tian_crispra": "Data.tian_crispra",
    "norman": "Data.norman",
    "dixit": "Data.dixit",
    "frangieh": "Data.Frangieh",
    "yao": "Data.Yao",
}

SINGLE_DATASET_KEYS = (
    "replogle_rpe1",
    "replogle_k562",
    "nadig_hepg2",
    "nadig_jurkat",
    "tian_crispri",
    "xu",
    "tian_crispra",
    "dixit",
    "frangieh",
)

DATASET_DISPLAY_NAMES = {
    "replogle_rpe1": "Replogle (RPE1)",
    "replogle_k562": "Replogle (K562)",
    "nadig_hepg2": "Nadig (HepG2)",
    "nadig_jurkat": "Nadig (Jurkat)",
    "tian_crispri": "Tian (CRISPRi)",
    "xu": "Xu",
    "tian_crispra": "Tian (CRISPRa)",
    "dixit": "Dixit",
    "frangieh": "Frangieh",
    "yao": "Yao (THP-1 KO)",
}


def _normalized_dataset_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(name)).casefold()
    return re.sub(r"[^a-z0-9]+", "", normalized)


_DATASET_ALIASES = {
    _normalized_dataset_name(alias): key
    for key in SINGLE_DATASET_KEYS
    for alias in (key, DATASET_DISPLAY_NAMES[key], key.replace("_", " "))
}


def resolve_single_dataset_key(name: str) -> str:
    key = _DATASET_ALIASES.get(_normalized_dataset_name(name))
    if key is None:
        allowed = ", ".join(DATASET_DISPLAY_NAMES[item] for item in SINGLE_DATASET_KEYS)
        raise ValueError(f"Unknown single dataset {name!r}. Allowed datasets: {allowed}")
    return key


def resolve_single_dataset_keys(names: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(resolve_single_dataset_key(name) for name in names))


def dataset_adapter(key: str):
    try:
        return import_module(DATASET_MODULES[key])
    except KeyError as exc:
        raise KeyError(f"Unknown dataset {key!r}; choose from {sorted(DATASET_MODULES)}") from exc


def load_dataset(
    key: str, source: Optional[Union[str, Path]] = None, *, biogrid_path: Optional[Path] = None,
) -> tuple[ad.AnnData, object]:
    module = dataset_adapter(key)
    if key == "yao":
        return module.load_raw(source, biogrid_path=biogrid_path), module.SPEC
    return module.load_raw(source), module.SPEC
