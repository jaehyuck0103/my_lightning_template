import math
import random
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

TRAIN_IMG_DIR = Path("Data") / "train_images"
TRAIN_CSV_PATH = Path("Data") / "train.csv"
TEST_IMG_DIR = Path("Data") / "test_images"


class Dataset1Cfg(BaseModel):
    name: Literal["dataset1"]
    mode: Literal["train"] | Literal["val"]
    epoch_scale_factor: float
    kfold_N: int
    kfold_I: int


def _imread_float(f):
    img = cv2.imread(f).astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

    return img


class Dataset1(Dataset):
    def __init__(self, cfg: Dataset1Cfg):
        super().__init__()

        self.mode = cfg.mode

        df = pd.read_csv(
            TRAIN_CSV_PATH, names=["basename", "label"], dtype={"basename": str, "label": int}
        )

        num_imgs = df.shape[0]
        kf = StratifiedKFold(n_splits=cfg.kfold_N, shuffle=True, random_state=910103)
        train_idx, valid_idx = list(kf.split(np.zeros(num_imgs), df["label"]))[cfg.kfold_I]

        self.df = df
        if cfg.mode == "train":
            self.idx_map = train_idx
        elif cfg.mode == "val":
            self.idx_map = valid_idx
        else:
            raise ValueError(f"Unknown Mode {cfg.mode}")

        #
        self.epoch_scale_factor = cfg.epoch_scale_factor

    def __getitem__(self, _idx):
        if self.epoch_scale_factor < 1:
            _idx += len(self) * random.randrange(math.ceil(1 / self.epoch_scale_factor))

        idx = self.idx_map[_idx % len(self.idx_map)]

        basename = self.df["basename"].iloc[idx]
        label = self.df["label"].iloc[idx]

        img_path = TRAIN_IMG_DIR / f"{basename}.jpg"
        img = _imread_float(str(img_path))

        img = cv2.resize(img, (224, 224))

        if self.mode == "train":
            # some augmentation
            pass
        else:
            # no augmentation
            pass

        # Add depth channels
        img = np.transpose(img, (2, 0, 1))
        # img = np.stack([img, img, img], axis=0)  # gray -> 3channel

        # sample return
        # label = np.expand_dims(label, axis=0)
        sample = {"img": img, "label": label}

        return sample

    def __len__(self):
        return round(len(self.idx_map) * self.epoch_scale_factor)
