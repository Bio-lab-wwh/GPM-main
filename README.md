# GPM · Network-guided genetic perturbation modeling

GPM predicts single-cell transcriptional responses to genetic perturbations
across cellular contexts. It models a perturbation as an intervention at a
target gene in a biological interaction network. The target gene's
context-conditioned state and the perturbation modality determine the
intervention signal. The state of each receiving gene then modulates its
response to that signal.

![GPM architecture showing control-derived context, biological interaction encoding, target interventions, and gene-specific responses](img/GPM.png)

**Figure 1. GPM architecture.** GPM combines control expression statistics
and a sample-level context representation with structurally pretrained gene
embeddings. Relation-specific aggregation over BioGRID physical and genetic
interactions and global linear attention yield graph-enhanced gene states.
Intervention signals reach all genes in the expression panel, with
gene-specific modulation at the receiving end. A shared decoder predicts
signed expression changes relative to the control population mean.

The implementation supports single- and multi-gene perturbations and encodes
knockdown, knockout, and overexpression as distinct modalities. Training
uses a group-mean squared error and a Sinkhorn distributional loss to align
predicted and observed perturbation groups.

## Installation

Use Python 3.12 in a virtual environment. From the repository root, create
and activate the environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate the environment with
`.\.venv\Scripts\Activate.ps1`.
The training configuration defaults to CUDA. For GPU training, first install
a PyTorch build compatible with your driver using the
[PyTorch installation guide](https://docs.pytorch.org/get-started/locally/).
Then install the project dependencies.

```bash
python -m pip install -r requirements.txt
```

Set `device` to `"cpu"` in the training configuration to select CPU execution.
The [requirements file](requirements.txt) lists the runtime dependencies.

## Data availability

No new datasets were generated for this study. All experiments use existing
public single-cell genetic perturbation datasets. BioGRID provides the
physical and genetic interaction priors.

| Resource | Input used by GPM | Public source |
| --- | --- | --- |
| BioGRID | Physical and genetic interaction prior | [BioGRID downloads](https://downloads.thebiogrid.org/BioGRID) |
| Replogle | K562 essential-gene and RPE1 CRISPRi screens | [Genome-Wide Perturb-seq data portal](https://gwps.wi.mit.edu/) |
| Nadig | HepG2 and Jurkat CRISPRi screens | [GEO GSE264667](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667) |
| Dixit | K562 knockout screen | [GEO GSE90063](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90063) |
| Norman | K562 single- and double-gene CRISPRa perturbations | [GEO GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344) |
| Xu | HEK293 CRISPRi screen | [GEO GSE218566](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE218566) |
| Yao | THP-1 conventional knockout screen | [GEO GSM6858447](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM6858447) |
| Tian | iPSC CRISPRi and CRISPRa screens | [scPerturb release on Zenodo](https://doi.org/10.5281/zenodo.13350497) |
| Frangieh | Patient-derived melanoma Perturb-CITE-seq | [Broad Single Cell Portal SCP1064](https://singlecell.broadinstitute.org/single_cell/study/SCP1064/multi-modal-pooled-perturb-cite-seq-screens-in-patient-models-define-novel-mechanisms-of-cancer-immune-evasion) |

The Replogle adapters expect `K562_essential_raw_singlecell_01.h5ad` and
`rpe1_raw_singlecell_01.h5ad` from the 2022 genome-scale Perturb-seq study.
The related [GEO GSE146194](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE146194)
record belongs to the earlier Replogle study.

See [the dataset adapters](Data/) for the expected per-dataset files and
metadata. Their default raw inputs are under `Data/`. Use the repeatable
`--source DATASET_KEY=RAW_PATH` option to override them. Gene Ontology (GO)
filtering is disabled by default. To match a GO-filtered baseline protocol,
use `--go-filter` with the [GEARS GO annotation mapping](https://dataverse.harvard.edu/api/access/datafile/6153417).
The default mapping path is `Data/gene2go_all.pkl` and can be overridden
with `--gene2go-path`.

## Running GPM

Place a human BioGRID interaction table at `GPM/gene_embedding/BIOGRID.tsv`
before preprocessing and training. Use a tab-separated export with official
symbol and synonym columns for both interactors and an experimental system
type column. From the repository root, train the BioGRID structural
embeddings, place `K562_essential_raw_singlecell_01.h5ad` under `Data/`,
and prepare the split and fit GPM.

```bash
python GPM/pre_train/train_embedding.py
python -m Division.single_unseen_pert --dataset replogle_k562
python -m Task.single_unseen_pert --dataset replogle_k562
```

The structural pretraining command saves `gene_embedding.pt` beside the
BioGRID table by default. For custom locations, set `biogrid_path` and
`gene_embedding_path` in the training JSON and pass it with `--train-config`.
Relative paths in the training JSON are resolved from the repository root.
Pass the same BioGRID path to the split script via `--biogrid-path`.
If the structural embedding file is missing, GPM initializes the gene
embeddings randomly.

The split script saves processed inputs and a manifest under
`Division/outputs/<task>/`. See [EXPERIMENTS.md](EXPERIMENTS.md) for
combinatorial perturbations, cross-cell-line transfer, double-zero-shot
prediction, joint training across perturbation modalities, and transfer to
patient-derived cells.

## Default model and training settings

These values are the defaults in the
[repository training configuration](GPM/gpm_train_config.json).
Structural pretraining has its own
[command-line configuration](GPM/pre_train/train_embedding.py).

| Component | Configuration key | Default | Description |
| --- | --- | ---: | --- |
| Model | `gene_embedding_dim` | 128 | Gene identity embedding dimension |
| Model | `cell_context_dim` | 256 | Global cellular context dimension |
| Model | `enhanced_node_dim` | 128 | Context-conditioned gene state and intervention dimension |
| Model | `technology_embedding_dim` | 128 | Perturbation modality embedding dimension |
| Execution | `edge_chunk_size` | 131,072 | Chunking budget for receiver-side modulation |
| Training | `seed` | 42 | Random seed |
| Training | `batch_size` | 240 | Training micro-batch size |
| Training | `pert_group_size` | 24 | Cells per perturbation supervision group |
| Training | `epochs` | 100 | Maximum epochs for standard training |
| Training | `min_groups` | 8,000 | Completed training groups required before checkpoint selection |
| Training | `patience` | 6 | Validation epochs without improvement before early stopping |
| AdamW | `lr` | 0.001 | Initial learning rate |
| AdamW | `lr_decay` | 0.98 | Per-epoch learning rate multiplier |
| AdamW | `weight_decay` | 0.001 | Weight decay coefficient |
| Training | `grad_clip_norm` | 1.0 | Maximum gradient norm |
| Loss | `loss_blur` | 0.02 | Entropic regularization scale for Sinkhorn transport |
| Loss | `loss_sinkhorn_iters` | 32 | Sinkhorn iterations |
| Loss | `loss_weights.expression_mse` | 1.0 | Group-mean expression MSE weight |
| Loss | `loss_weights.ot` | 0.01 | Sinkhorn distributional loss weight |

## Citation

The bibliographic citation and DOI for GPM will be provided after formal
publication.

## Acknowledgements

We thank the teams that generated and shared the public perturbation datasets,
the BioGRID curators for maintaining the interaction resource, and NCBI GEO,
the Genome-Wide Perturb-seq portal, Zenodo, scPerturb, and the Broad Single
Cell Portal for making these data accessible. We also acknowledge the GEARS
developers for providing the Gene Ontology annotation mapping.
