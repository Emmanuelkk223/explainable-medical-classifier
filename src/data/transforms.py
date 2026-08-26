import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from typing import List, Tuple, Dict, Any

# Standard ImageNet normalization parameters
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_image_transforms(img_size: int = 224) -> Tuple[A.Compose, A.Compose]:
    """
    Returns train and evaluation Albumentations transform pipelines.
    Includes spatial, color, and cutout augmentations for training.
    """
    train_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5
            ),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4
            ),
            # Simulates visual occlusions like hair/air bubbles on dermoscopic lenses
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill_value=0,
                p=0.3,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    return train_transform, val_transform


def build_tabular_pipeline(
    numerical_cols: List[str], categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Builds a Scikit-Learn ColumnTransformer for tabular preprocessing.
    Must be fit ONLY on training fold data.
    """
    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols),
        ]
    )
    return preprocessor
