# GPM experiments

Run the commands in this guide from the repository root. The `Data/` adapters
load raw datasets. The `Division/` scripts preprocess the data and save the
processed H5AD files, control means, and split metadata under
`Division/outputs/<task>/` by default. The corresponding `Task/` scripts read
`manifest.json` from that directory and train and evaluate the model.

Use `--source DATASET_KEY=RAW_PATH` to supply a local dataset path. Repeat the
option for multiple datasets. The default raw input directory is `Data/`.
To use a custom BioGRID path, pass `--biogrid-path` to the split script and
set `biogrid_path` in the training JSON. Pass that JSON to the training
script with `--train-config`.

## Preprocessing

Preprocessing requires nonnegative, approximately integer count matrices.
The default split scripts retain perturbation conditions with strictly more
than 50 cells. To restrict targets to GO-annotated genes for a baseline
comparison, pass `--go-filter`. The mapping defaults to
`Data/gene2go_all.pkl` and can be supplied with `--gene2go-path`.
BioGRID supports gene-symbol mapping and provides the model's graph prior.
Genes outside BioGRID remain eligible for the expression panel.

Each panel combines 2,000 highly variable genes (HVGs), selected from the
relevant target controls, with the retained perturbation targets. For
multi-fold cross-cell-line benchmarks, the saved H5AD and control means
contain the union of the fold panels. Training, inference, and evaluation
use the ordered panel recorded in the manifest for the current fold.

## Unseen perturbations within a dataset

`single_unseen_pert` supports Replogle (RPE1), Replogle (K562), Nadig (HepG2),
Nadig (Jurkat), Tian (CRISPRi), Xu, Tian (CRISPRa), Dixit, and Frangieh.
Each dataset is preprocessed and split independently. The split holds
out approximately 30% of perturbation conditions for testing and uses
10% of the remaining conditions for validation. HVG selection uses the
dataset's control cells.

```bash
python -m Division.single_unseen_pert "Replogle (RPE1)" "Nadig (HepG2)" Xu
python -m Task.single_unseen_pert "Replogle (RPE1)" "Nadig (HepG2)" Xu
```

Dataset keys can also be passed with the repeatable `--dataset` option.

```bash
python -m Division.single_unseen_pert \
  --dataset replogle_rpe1 --dataset nadig_hepg2 --dataset xu
```

Both scripts default to the two Replogle datasets. Set `DEFAULT_DATASETS`
to an empty list in the training entry point to run all manifest folds
without specifying datasets. The training script lists the available
datasets if a requested dataset is missing.

The split script reuses complete, compatible dataset artifacts and appends new
datasets to the existing manifest. Each run refreshes `split_summary.csv`
and `split_summary.json` with perturbation and cell counts for all saved folds.

## Other benchmarks

| Task | Design |
| --- | --- |
| `double_unseen_pert` | Evaluates Norman single-gene perturbations and combinations with zero, one, or two unseen targets. |
| `unseen_cell` | Holds out each of four knockdown cell lines in turn. HVG selection uses controls from the held-out cell line. |
| `unseen_pert_cell` | Runs six double-zero-shot folds across six knockdown datasets. Test perturbations are absent from all source datasets in each fold. HVG selection uses controls from the held-out cell line. |
| `across_tech` | Jointly trains on Tian CRISPRi and CRISPRa. The panel uses the shared measured genes and controls from both datasets for HVG selection. Perturbation conditions are split globally across the two technologies. |

Run `python -m Division.<task>` to prepare a benchmark and
`python -m Task.<task>` to run it. For the cross-cell-line benchmarks,
GPM uses held-out controls for feature selection and inference. Parameter
updates use the source data.

## Transfer to patient-derived cells

`Division/zero_patient.py` prepares six source datasets and the Frangieh
target dataset. The source pool contains Dixit, Xu, Yao (THP-1 KO),
Tian CRISPRi, Tian CRISPRa, and Norman single-gene perturbations.
Ten percent of the source perturbation conditions are held out for
pretraining validation. The panel uses genes measured across source and
target datasets, with HVGs selected from Frangieh controls.

Frangieh perturbation conditions are split into disjoint fine-tuning (70%)
and test (30%) sets. The test set can include perturbations observed in the
source datasets. Frangieh controls are assigned to the `inference` split and
provide basal expression inputs and the control reference for effect
calculation. Zero-shot evaluation precedes all target-domain parameter updates.

```bash
python -m Division.zero_patient
python -m Task.zero_patient
```

The task saves the source-pretrained checkpoint and zero-shot predictions,
then fine-tunes for 10 epochs and evaluates the held-out test set after each
epoch. It writes 11 per-stage metric JSON files and saves the fine-tuned
checkpoint and predictions at epoch 10.
