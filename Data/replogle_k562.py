"""Raw Replogle K562 essential-gene CRISPRi adapter."""

from pathlib import Path
from typing import Optional, Union
from Data.preprocessing import DatasetSpec, DATA_DIR, load_h5ad_dataset

SPEC = DatasetSpec(
    key="replogle_k562", dataset_id="Replogle2022_K562_KD", cell_line="K562",
    technology="KD", default_source=DATA_DIR / "K562_essential_raw_singlecell_01.h5ad",
    perturbation_columns=("gene", "target_gene", "perturbation", "condition"),
    batch_columns=("gem_group", "batch", "batch_id", "replicate", "library"),
    control_tokens=("non-targeting", "non_targeting"),
)

def load_raw(source: Optional[Union[str, Path]] = None):
    return load_h5ad_dataset(SPEC, source)
