from typing import Literal

from pydantic import BaseModel, Field

from .dataset1 import Dataset1, Dataset1Cfg


class Dataset2Cfg(BaseModel):
    name: Literal["dataset2"]


class DatasetCfg(BaseModel):
    specific: Dataset1Cfg | Dataset2Cfg = Field(..., discriminator="name")


def get_dataset(cfg: DatasetCfg, mode: str):
    if cfg.specific.name == "dataset1":
        dataset = Dataset1(cfg.specific, mode=mode)
        print(f"{len(dataset)} items\n")
    else:
        raise ValueError(f"Unexpected Dataset: name={cfg.specific.name}")
    return dataset
