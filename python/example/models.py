from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl


class MNISTClassifier(pl.LightningModule):

    def __init__(self, **kwargs):
        super(MNISTClassifier, self).__init__()

        self.save_hyperparameters(
            "feat_out1", "feat_out2",
            "feat_out3", "clf_hid",
            "feat_lr", "clf_lr",
            "batch_size")

        self.args = kwargs

        self.feature = nn.Sequential(
            nn.Conv2d(1, kwargs["feat_out1"], 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.LazyConv2d(kwargs["feat_out2"], 3),
            nn.MaxPool2d(2, 2),

            nn.LazyConv2d(kwargs["feat_out3"], 3),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(kwargs["clf_hid"]),
            nn.ReLU(True),
            nn.LazyLinear(10)
        )

        self.criterion = nn.CrossEntropyLoss()

        # self.automatic_optimization = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        group = parser.add_argument_group("MNISTClassifier")

        for i, size in enumerate([8, 16, 8], 1):
            group.add_argument(
                f"--feat_out{i}", type=int, default=size, metavar="N",
                help=f"output channel size for feature layer {i} (default: {size})")

        group.add_argument(
            "--clf_hid", type=int, default=32, metavar="N",
            help="hidden size for classifier layer (default: 32)")

        for layer, lr in [('feat', 1e-2), ('clf', 1e-3)]:
            group.add_argument(
                f"--{layer}_lr", type=float, default=lr,
                metavar="LR",
                help=f"learning rate (default: {lr})")

        return parser

    def configure_optimizers(self):
        optimizer = optim.SGD([
            {'params': self.feature.parameters(), 'lr': self.args["feat_lr"]},
            {'params': self.classifier.parameters(), 'lr': self.args["clf_lr"]}
        ], momentum=0.9)

        return optimizer

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        input, target = batch       # no .to(device) required
        output = self(input)
        loss = self.criterion(output, target)

        return {"loss": loss}

    # def training_step(self, batch, batch_idx):
    #     """
    #     If self.automatic_optimization = False in self.__init__()
    #     see https://pytorch-lightning.readthedocs.io/en/latest/common/optimization.html
    #     """
    #     opt = self.optimizers()
    #     opt.zero_grad()

    #     input, target = batch
    #     output = self(input)
    #     loss = self.criterion(output, target)

    #     self.manual_backward(loss)
    #     opt.step()

    #     return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", avg_loss)

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self(input)
        loss = self.criterion(output, target)
        _, prediction = torch.max(output, 1)
        correct = float((prediction == target).sum())

        return {"val_step_loss": loss,
                "num_correct": correct,
                "num_samples": target.size(0)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_step_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)

        correct = sum([x["num_correct"] for x in outputs])
        total = sum(x["num_samples"] for x in outputs)
        self.log("val_acc", correct / total)

    def test_step(self, batch, batch_idx):
        input, target = batch
        output = self(input)
        _, prediction = torch.max(output, 1)
        correct = float((prediction == target).sum())

        return {"num_correct": correct,
                "num_samples": target.size(0)}

    def test_epoch_end(self, outputs):
        correct = sum(x["num_correct"] for x in outputs)
        total = sum(x["num_samples"] for x in outputs)
        self.log("test_acc", correct / total)
