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
    
    def train_dataloader(self):
        dataset = load_dataset(name=self.data_name,
                                data_dir=self.data_dir,
                                fold=self.fold,
                                split='train')
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=dataset.my_collate
        )
        return dataloader
    
    def val_dataloader(self):
        dataset = load_dataset(name=self.data_name,
                                data_dir=self.data_dir,
                                fold=self.fold,
                                split='val')
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=dataset.my_collate
        )
        return dataloader
    
    def test_dataloader(self):
        dataset = load_dataset(name=self.data_name,
                                data_dir=self.data_dir,
                                fold=self.fold,
                                split='test')
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=dataset.my_collate
        )
        return dataloader