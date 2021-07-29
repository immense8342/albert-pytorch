from typing import Text, Dict, Optional

import torch


class Config(object):
    """
   Construct an ALBERT config.

   Args:
       num_classes (:obj:`str`):
           the num of classes
   """


    def __init__(
            self,
            num_classes: Optional[int] = 3,
            batch_size: Optional[int] = 32,
            learning_rate: Optional[float] = 3e-5,
            epochs: Optional[int] = 20,
            max_length: Optional[int] = 100,
            train_data: Optional[Text] = "data/text-classifizer/train.csv",
            eval_data: Optional[Text] = "data/text-classifizer/dev.csv",
            albert_vocab_file: Optional[Text] = "albert_base_zh/vocab_chinese.txt",
            albert_hidden_size: Optional[int] = 768,
            albert_pytorch_model_path: Optional[Text] = "models"

    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.num_classes = num_classes
        self.train_data = train_data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_length = max_length
        self.albert_vocab_file = albert_vocab_file
        self.albert_hidden_size = albert_hidden_size
        self.albert_pytorch_model_path = albert_pytorch_model_path
