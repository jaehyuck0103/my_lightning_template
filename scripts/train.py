from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import tomli
import typer
from pydantic import BaseModel, StrictInt
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils import data

from project_name.datasets import DatasetCfg, get_dataset
from project_name.lightning_modules.callbacks.my_printing_callback import (
    MyPrintingCallback,
)
from project_name.lightning_modules.callbacks.scalar_tb_callback import (
    ScalarTensorboardCallback,
)
from project_name.lightning_modules.classification import (
    PlClassification,
    PlClassificationCfg,
)


class TrainCfg(BaseModel):
    train_batch_size: StrictInt
    val_batch_size: StrictInt
    dataset: DatasetCfg
    pl_module: PlClassificationCfg


def main(config_path: Path):

    cfg = TrainCfg.parse_obj(tomli.loads(config_path.read_text("utf-8")))
    log_dir = Path("Logs") / config_path.stem / datetime.now().strftime("%y%m%d_%H%M%S")

    #
    train_dataset = get_dataset(cfg.dataset, mode="train")
    val_dataset = get_dataset(cfg.dataset, mode="val")

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    trainer = pl.Trainer(
        default_root_dir=log_dir,
        accelerator="cuda",
        devices=1,
        max_epochs=cfg.pl_module.optim.lr_milestones[-1],
        benchmark=True,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, every_n_epochs=1),
            MyPrintingCallback(),
            ScalarTensorboardCallback(),
        ],
        enable_progress_bar=False,
        logger=False,
    )

    model = PlClassification(cfg.pl_module)
    trainer.fit(model, train_loader, [val_loader, val_loader])

    """
    # Load best checkpoint after training
    model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, cfg=cfg)

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    print("ViT results", result)
    """


if __name__ == "__main__":
    typer.run(main)