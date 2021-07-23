import collections
import os
import re
from typing import Text, Optional, Dict, Set
import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomTextClassifizerDataset(Dataset):
    """classifizer dataset"""

    def __init__(self, filepath, tokenizer, max_length):
        self.dataframe = pd.read_csv(filepath)
        self.text_dir = filepath
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        labels = self.dataframe.iloc[idx, 0]
        text = self.dataframe.iloc[idx, 1]
        token = self.tokenizer(text, return_tensors='pt', padding="max_length", max_length=self.max_length,
                               truncation=True)
        item = {"labels": torch.tensor(labels, dtype=torch.long), "token": token}
        return item

    @property
    def num_classes(self) -> int:
        return len(set(self.dataframe["label"]))


