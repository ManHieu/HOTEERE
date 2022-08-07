from torch.utils.data import DataLoader 
import pytorch_lightning as pl

from arguments import DataTrainingArguments
from data_modules.datasets import load_dataset


class EEREDataModule(pl.LightningDataModule):
    """
    Dataset processing for Event Event Relation Extraction.
    """
    def __init__(self,
                data_name: str,
                batch_size: int,
                data_dir: str,
                fold: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.data_name = data_name
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.fold = fold
        self.dataset = load_dataset(name=self.data_name,
                                    data_dir=self.data_dir)
    
    def train_dataloader(self):
        dataloader = DataLoader(
            dataset=self.dataset.load_split(fold=self.fold, split='train'),
            batch_size=self.batch_size,
            shuffle=True
        )
        return dataloader
    
    def val_dataloader(self):
        dataloader = DataLoader(
            dataset=self.dataset.load_split(fold=self.fold, split='val'),
            batch_size=self.batch_size,
            shuffle=True
        )
        return dataloader
    
    def test_dataloader(self):
        dataloader = DataLoader(
            dataset=self.dataset.load_split(fold=self.fold, split='test'),
            batch_size=self.batch_size,
            shuffle=True
        )
        return dataloader