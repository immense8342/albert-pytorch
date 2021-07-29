import math

import torch
from typing import Optional, Text, List, Tuple

from torch import Tensor
from torch.autograd import Variable
from transformers import BertTokenizer, BertModel, AlbertModel

from config import Config


class TextClassificationModel(torch.nn.Module):
    """ rnn text classification"""

    def __init__(self, config: Optional[Config]):
        super(TextClassificationModel, self).__init__()
        self.max_length = config.max_length
        self.num_class = config.num_classes
        self.albert_hidden_size = config.albert_hidden_size
        self.albert = AlbertModel.from_pretrained(config.albert_pytorch_model_path)
        self.classifier = torch.nn.Linear(self.albert_hidden_size, config.num_classes)

    def forward(self, input_ids: Optional[Tensor], attention_mask: Optional[Tensor],
                token_type_ids: Optional[Tensor]) -> Tensor:
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return self.classifier(outputs.pooler_output)

    def summuary(self):
        print("Model structure: ", self, "\n\n")

        for name, param in self.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
