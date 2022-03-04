from argparse import ArgumentParser
import os
import random

import mlflow.pytorch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torchvision.utils import make_grid

from datamodules import MNISTDataModule
from models import MNISTClassifier


def parse_args():
    parser = ArgumentParser(description="MNIST Example")
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = MNISTClassifier.add_model_specific_args(
        parent_parser=parser)
    
    dm_group = parser.add_argument_group("MNISTDataModule")
    dm_group.add_argument(
        "--data_root", 
        type=str, 
        default="~/Datasets",
        metavar="PATH", help="MNIST root path")
    dm_group.add_argument(
        "--batch_size", 
        type=int, 
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)")
    dm_group.add_argument(
        "--val_batch_size", 
        type=int, 
        default=128,
        metavar="N",
        help="input batch size for validation (default: 128)")
    
    parser.add_argument(
        "--seed",
        type=int,
        default=777,
        metavar="N",
        help="random seed"
    )

    args = parser.parse_args()
    return args


def init_experiment(args):
    """
    For reproducibility, see https://pytorch.org/docs/stable/notes/randomness.html
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    

def train_evaluation(mlrun, args, hparams, log_samples=False):

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
        args, 
        default_root_dir=artifact_path, 
        callbacks=[mcp_callback],
        gpus=str(hash(os.getlogin()) % 4) if torch.cuda.is_available() else None,
        accelerator="gpu", devices=1
    )
    
    # inject mlflow auto logging
    mlflow.pytorch.autolog(log_models=False)
    mlflow.log_params(model.hparams)
    
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm)
    test_acc = trainer.callback_metrics.get("test_acc")
    
    if log_samples:
        mlflow_log_samples(dm)
    
    return float(test_acc)


def mlflow_log_samples(dm):

    data_iter = iter(dm.train_dataloader())
    input, _ = data_iter.next()

    img = make_grid(input, nrow=8)
    img = img * dm.std + dm.mean
    npimg = img.numpy()
    mlflow.log_image(npimg.transpose((1, 2, 0)),  
                     "samples.png")

def main():
    args = parse_args()
    init_experiment(args)
    
    hparams = dict(args.__dict__)
    del hparams["data_root"]
    del hparams["val_batch_size"]

    with mlflow.start_run() as mlrun:
        train_evaluation(mlrun, args, hparams, log_samples=True)
        

if __name__ == "__main__":
    main()

    