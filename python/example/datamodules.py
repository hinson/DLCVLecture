from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_root=None, batch_size=64,
                 val_batch_size=128, **kwargs):
        super(MNISTDataModule, self).__init__()

        self.data_root = data_root
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        self.mean = 0.1307
        self.std = 0.3081
        normalization = transforms.Normalize((self.mean,), (self.std,))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(5),
            normalization
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalization
        ])

    def prepare_data(self):
        # download
        MNIST(self.data_root, train=True, download=True)
        MNIST(self.data_root, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_set = MNIST(self.data_root, train=True,
                              transform=self.transform)
            self.train_set, self.val_set = random_split(
                train_set, [50000, 10000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = MNIST(self.data_root, train=False,
                                  transform=self.test_transform)

        if stage == "predict" or stage is None:
            self.mnist_predict = MNIST(self.data_root, train=False,
                                       transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.val_batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.val_batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict,
                          batch_size=self.val_batch_size)
