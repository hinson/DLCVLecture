from argparse import ArgumentParser
import os

import mlflow.pytorch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from datamodules import MNISTDataModule
from models import MNISTClassifier


def parse_args():
    parser = ArgumentParser(description="MNIST Example")
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = MNISTClassifier.add_model_specific_args(
        parent_parser=parser)
    
    dm_group = parser.add_argument_group("MNISTDataModule")
    dm_group.add_argument(
        "--data_root", type=str, default="~/Datasets",
        metavar="PATH", help="MNIST root path")
    dm_group.add_argument(
        "--batch_size", type=int, default=64, metavar="N",
        help="input batch size for training (default: 64)")
    dm_group.add_argument(
        "--val_batch_size", type=int, default=128, metavar="N",
        help="input batch size for validation (default: 128)")

    args = parser.parse_args()
    return args


def train_evaluation(mlrun, args, hparams):

    model = MNISTClassifier(**hparams)
    dm = MNISTDataModule(args.data_root, 
                         val_batch_size=args.val_batch_size, **hparams)
    
    artifact_path = mlrun.info.artifact_uri
    
    mcp_callback = ModelCheckpoint(
        dirpath=artifact_path+'/checkpoints',
        filename='{epoch}-{val_loss:.4f}-{val_acc:.4f}',
        save_top_k=1, monitor="val_loss", mode="min",
        save_weights_only=True
    )
    
    trainer = pl.Trainer.from_argparse_args(
        args, default_root_dir=artifact_path, callbacks=[mcp_callback],
        gpus=str(hash(os.getlogin()) % 4) if torch.cuda.is_available() else None
    )
    
    # inject mlflow auto logging
    mlflow.pytorch.autolog(log_models=False)
    mlflow.log_params(model.hparams)
    
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm)
    test_acc = trainer.callback_metrics.get("test_acc")
    
    return float(test_acc)


def main():
    args = parse_args()
    hparams = dict(args.__dict__)
    del hparams["data_root"]
    del hparams["val_batch_size"]

    with mlflow.start_run() as mlrun:
        train_evaluation(mlrun, args, hparams)
        

if __name__ == "__main__":
    main()

    