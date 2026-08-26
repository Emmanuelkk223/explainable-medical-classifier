import os
import cv2
import torch
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional, List, Tuple, Dict, Union, Any
from src.data.transforms import get_image_transforms, build_tabular_pipeline


class ISICMultimodalDataset(Dataset):
    """
    PyTorch Dataset returning dermoscopic image tensor, preprocessed
    tabular feature vector, and binary target label.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir_or_hdf5: str,
        tabular_features: np.ndarray,
        transform: Optional[Any] = None,
        is_hdf5: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.image_path_or_hdf5 = image_dir_or_hdf5
        self.tabular_features = torch.tensor(tabular_features, dtype=torch.float32)
        self.transform = transform
        self.is_hdf5 = is_hdf5
        self.isic_ids = self.df["isic_id"].values

        if "target" in self.df.columns:
            self.targets = torch.tensor(self.df["target"].values, dtype=torch.float32)
        else:
            self.targets = torch.zeros(len(self.df), dtype=torch.float32)

        self.hdf5_file = None

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, isic_id: str) -> np.ndarray:
        if self.is_hdf5:
            if self.hdf5_file is None:
                self.hdf5_file = h5py.File(self.image_path_or_hdf5, "r")
            img_bytes = self.hdf5_file[isic_id][()]
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        else:
            img_path = os.path.join(self.image_path_or_hdf5, f"{isic_id}.jpg")
            img = cv2.imread(img_path)
            if img is None:
                # Fallback synthetic matrix if image missing
                img = np.zeros((224, 224, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        isic_id = self.isic_ids[idx]
        image_np = self._load_image(isic_id)

        if self.transform is not None:
            augmented = self.transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.tensor(image_np).permute(2, 0, 1).float() / 255.0

        tabular_tensor = self.tabular_features[idx]
        target_tensor = self.targets[idx].unsqueeze(-1)

        return {
            "image": image_tensor,
            "tabular": tabular_tensor,
            "target": target_tensor,
            "isic_id": isic_id,
        }


def create_multimodal_dataloaders(
    metadata_splits_path: str,
    image_dir_or_hdf5: str,
    val_fold: int = 0,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Constructs leakage-free Train, Validation, and Test DataLoaders.
    Fits tabular preprocessing ONLY on the training fold.
    """
    df = pd.read_csv(metadata_splits_path)

    numerical_cols = ["age_approx", "clin_size_long_diam_mm"]
    categorical_cols = ["sex", "anatom_site_general"]

    # Filter columns to available metadata
    numerical_cols = [c for c in numerical_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    # Split dataset into train, val, test subsets
    train_df = df[~df["fold"].isin([val_fold, -2])].copy()
    val_df = df[df["fold"] == val_fold].copy()
    test_df = df[df["fold"] == -2].copy()

    # Fit tabular pipeline strictly on train fold
    tabular_pipeline = build_tabular_pipeline(numerical_cols, categorical_cols)
    X_train_tab = tabular_pipeline.fit_transform(train_df)
    X_val_tab = tabular_pipeline.transform(val_df)
    X_test_tab = tabular_pipeline.transform(test_df)

    tabular_dim = X_train_tab.shape[1]

    train_transform, val_transform = get_image_transforms()
    is_hdf5 = image_dir_or_hdf5.endswith(".hdf5") or image_dir_or_hdf5.endswith(".h5")

    train_dataset = ISICMultimodalDataset(
        train_df, image_dir_or_hdf5, X_train_tab, train_transform, is_hdf5
    )
    val_dataset = ISICMultimodalDataset(
        val_df, image_dir_or_hdf5, X_val_tab, val_transform, is_hdf5
    )
    test_dataset = ISICMultimodalDataset(
        test_df, image_dir_or_hdf5, X_test_tab, val_transform, is_hdf5
    )

    sampler = None
    if use_weighted_sampler and "target" in train_df.columns:
        targets = train_df["target"].values
        class_counts = np.bincount(targets.astype(int))
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[targets.astype(int)]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, tabular_dim
