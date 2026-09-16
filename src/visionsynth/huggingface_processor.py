import datasets
import pandas as pd


def load_huggingface_dataset(
    dataset_name: str,
    config_name: str | None = None,
    split: str | None = None,
    streaming: bool = False,
) -> pd.DataFrame | datasets.IterableDataset | None:
    """Load dataset using the datasets library"""
    try:
        dataset = datasets.load_dataset(
            dataset_name,
            config_name,
            split=split,
            streaming=streaming,
        )

        if streaming:
            return dataset

        # Handle different dataset structures
        if isinstance(dataset, datasets.DatasetDict):
            # If multiple splits, use the first one or 'train' if available
            if split:
                return dataset[split].to_pandas()
            if "train" in dataset:
                return dataset["train"].to_pandas()
            # Use the first available split
            first_split = next(iter(dataset.keys()))
            return dataset[first_split].to_pandas()

        return dataset.to_pandas()

    except Exception:
        return None


def push_output_to_hub(
    output_dir: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Add generated synthetic dataset",
) -> None:
    """Upload a generated output directory (images + CSV) to the Hub as a dataset repo.

    Requires the caller to already be authenticated (`huggingface-cli login` or
    the `HF_TOKEN` env var) — this function does not accept or handle a token directly.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )
