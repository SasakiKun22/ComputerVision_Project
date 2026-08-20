from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

subset_fraction: float = 1.0
seed: int = 42

class NIGHTSDataset(Dataset):
    """
    PyTorch Dataset for the NIGHTS perceptual similarity dataset.

    Each sample contains:
        - reference image
        - left candidate image
        - right candidate image
        - target:
            0 -> LEFT is preferred
            1 -> RIGHT is preferred
        - sample id

    Following the official DreamSim implementation,
    only triplets with at least `min_votes` votes are kept.
    """

    VALID_SPLITS = {"train", "val", "test"}

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        min_votes: int = 6,
        subset_fraction: float = 1.0,
        seed: int = 42,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.min_votes = min_votes
        self.subset_fraction = subset_fraction
        self.seed = seed

        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Expected one of {sorted(self.VALID_SPLITS)}."
            )

        if not 0 < subset_fraction <= 1.0:
            raise ValueError(
                "subset_fraction must be in the interval (0, 1]."
            )

        csv_path = self.root_dir / "data.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Could not find NIGHTS metadata file: {csv_path}"
            )

        self.data = pd.read_csv(csv_path)

        required_columns = {
            "id",
            "left_vote",
            "right_vote",
            "votes",
            "ref_path",
            "left_path",
            "right_path",
            "split",
        }

        missing_columns = required_columns - set(self.data.columns)

        if missing_columns:
            raise ValueError(
                f"Missing columns in data.csv: {missing_columns}"
            )

        # Keep only sufficiently reliable human judgments.
        self.data = self.data[
            self.data["votes"] >= self.min_votes
        ]

        # Select requested official split.
        self.data = self.data[
            self.data["split"] == self.split
        ]

        # Deterministic manageable subset.
        if self.subset_fraction < 1.0:
            self.data = self.data.sample(
                frac=self.subset_fraction,
                random_state=self.seed,
            )

        self.data = self.data.reset_index(drop=True)

        if len(self.data) == 0:
            raise RuntimeError(
                f"No samples found for split '{split}'."
            )

    def __len__(self):
        return len(self.data)

    def _load_rgb(self, relative_path):
        image_path = self.root_dir / relative_path

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __getitem__(self, index):
        row = self.data.iloc[index]

        sample_id = int(row["id"])

        # DreamSim uses right_vote as the 2AFC target:
        # 0 -> LEFT preferred
        # 1 -> RIGHT preferred
        target = float(row["right_vote"])

        if target not in (0.0, 1.0):
            raise ValueError(
                f"Invalid right_vote={target} "
                f"for sample {sample_id}"
            )

        reference = self._load_rgb(row["ref_path"])
        left = self._load_rgb(row["left_path"])
        right = self._load_rgb(row["right_path"])

        return {
            "reference": reference,
            "left": left,
            "right": right,
            "target": target,
            "id": sample_id,
        }