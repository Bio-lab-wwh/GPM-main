from Division.common import run_cli

# Set datasets for runs without arguments. An empty list selects both Replogle datasets.
DEFAULT_DATASETS = ["Replogle (RPE1)","Replogle (K562)"]

if __name__ == "__main__":
    run_cli(
        "single_unseen_pert",
        allow_dataset_selection=True,
        default_datasets=DEFAULT_DATASETS,
    )
