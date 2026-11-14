"""Raw Nadig Jurkat CRISPRi adapter."""

from pathlib import Path
from typing import Optional, Union
from Data.preprocessing import DatasetSpec, DATA_DIR, load_h5ad_dataset

SPEC = DatasetSpec(
    key="nadig_jurkat", dataset_id="Nadig2024_Jurkat_KD", cell_line="Jurkat",
    technology="KD", default_source=DATA_DIR / "GSE264667_jurkat_raw_singlecell_01.h5ad",
    perturbation_columns=("gene", "target_gene", "perturbation", "condition", "target"),
    batch_columns=("gem_group", "batch", "batch_id", "replicate", "library"),
    control_tokens=("non-targeting", "non_targeting"),
)

def load_raw(source: Optional[Union[str, Path]] = None):
    return load_h5ad_dataset(SPEC, source)
