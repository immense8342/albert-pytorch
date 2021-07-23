import time
import torch
from typing import Optional, Text
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AlbertTokenizerFast, AlbertConfig, AlbertForPreTraining, \
    load_tf_weights_in_albert, PreTrainedTokenizer, AlbertModel
from tokenization import AlBertTokenizer
from utils import CustomTextClassifizerDataset
from trainer import TextClassifizerTrainer
from model import TextClassificationModel
from config import Config

# load config from object
config = Config()

# tokenizer
tokenizer = AlBertTokenizer(config.albert_vocab_file)

# load dataset
train_datasets = CustomTextClassifizerDataset(config.train_data, tokenizer, config.max_length)
eval_datasets = CustomTextClassifizerDataset(config.eval_data, tokenizer, config.max_length)

# dataloader
train_dataloader = DataLoader(train_datasets, batch_size=config.batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_datasets, batch_size=config.batch_size, shuffle=True)

# create model
model = TextClassificationModel(config)
# create trainer
trainer = TextClassifizerTrainer(
    model=model,
    args=None,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    epochs=config.epochs,
    learning_rate=config.learning_rate,
    device=config.device
)

# train model
trainer.train()
