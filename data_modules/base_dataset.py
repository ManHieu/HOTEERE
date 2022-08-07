from abc import ABC, abstractmethod
import logging
import math
import os
import pickle
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from data_modules.input_example import InputExample, InputFeature


class BaseDataset(Dataset, ABC):
    """
    Base class for all datasets.
    """
    name = None         # name of the dataset


    def __init__(
        self,
        data_dir: str,
    ):
        self.data_path = data_dir
        self.examples = self.load_data(self.data_path)
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, index):
        return self.features[index]

    def load_split(self, fold=0, split='train'):
        split_corpus = self.examples[fold][split]
        self.features = self.compute_features(split_corpus)
        self.size = len(self.features)
        return self

    @abstractmethod
    def load_schema(self):
        """
        Load extra dataset information, such as entity/relation types.
        """
        pass

    @abstractmethod
    def load_data(self, split: str, data_path: str) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        pass

    def compute_features(self, examples: List[InputExample]) :
        """
        Compute features for model 
        """
        return examples

        
