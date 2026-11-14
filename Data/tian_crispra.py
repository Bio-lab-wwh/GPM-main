"""Raw Tian iPSC CRISPRa adapter."""

from pathlib import Path
from typing import Optional, Union
from Data.preprocessing import DatasetSpec, DATA_DIR, load_h5ad_dataset

SPEC = DatasetSpec(
    key="tian_crispra", dataset_id="TianKampmann2021_iPSC_CRISPRa", cell_line="iPSC",
    technology="OE", default_source=DATA_DIR / "TianKampmann2021_CRISPRa.h5ad",
    perturbation_columns=("perturbation", "condition", "target", "gene", "target_gene"),
    batch_columns=("batch", "batch_id", "replicate", "sample", "library"),
    control_tokens=("non-targeting", "non_targeting"),
)

def load_raw(source: Optional[Union[str, Path]] = None):
    return load_h5ad_dataset(SPEC, source)
