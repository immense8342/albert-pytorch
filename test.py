import torch
from transformers import BertTokenizer, AlbertModel
from config import Config
from model import TextClassificationModel
import torch.nn.functional as F
import pandas as pd

# tokenizer
from tokenization import AlBertTokenizer

# load config from object
config = Config()

# tokenizer
tokenizer = AlBertTokenizer(config.albert_vocab_file)
model = TextClassificationModel(config)

model.load_state_dict(torch.load('models/text_classifizer-checkpoint_8_epoch.pkl')["model_state_dict"])

model.to(config.device)
model.eval()
with torch.no_grad():
    dataframe = pd.read_csv("data/text-classifizer/test.csv")
    text_list = dataframe["text"]
    label = dataframe["label"]
    for idx, text in enumerate(text_list):
        token = tokenizer(text, return_tensors='pt', padding="max_length", max_length=config.max_length,
                          truncation=True)
        input_ids = token["input_ids"].squeeze(1).to(config.device)
        attention_mask = token["attention_mask"].squeeze(1).to(config.device)
        token_type_ids = token["token_type_ids"].squeeze(1).to(config.device)

        # Compute prediction and loss
        pred = model(input_ids, attention_mask, token_type_ids)
        scores = F.softmax(pred)
        pred = pred.argmax(1).item()
        if pred == label[idx]:
            print(f'correct{pred} {label[idx]},scores:{scores}{text}')
        else:
            print(f'error{pred} {label[idx]},scores:{scores}{text}')
