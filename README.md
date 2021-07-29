# ALBERT 中文文本分类 (微调albert)

基于ALBERT的中文情感分类任务

#### 文本分类

bert-dense

```
这里的文本分类主要是多分类，如果是二分类任务可以自己替换损失函数
``` 

#### google 发布的中文预训练模型
Chinese models are released. We would like to thank [CLUE team ](https://github.com/CLUEbenchmark/CLUE) for providing the training data.
- [Base](https://storage.googleapis.com/albert_models/albert_base_zh.tar.gz)
- [Large](https://storage.googleapis.com/albert_models/albert_large_zh.tar.gz)
- [Xlarge](https://storage.googleapis.com/albert_models/albert_xlarge_zh.tar.gz)
- [Xxlarge](https://storage.googleapis.com/albert_models/albert_xxlarge_zh.tar.gz)



#### tensorflow预训练模型转换为pytorch模型

```
python  convert_pytorch_model.py  \
    --tf_checkpoint ./<ALBERT_PRETRAIN_MODEL_PATH>/model.ckpt-best  \
    --albert_config_file ./<ALBERT_PRETRAIN_MODEL_PATH>/albert_config.json  \
    --pytorch_dump_path models/pytorch_pretrain_model.bin
```

##### 环境配置

```
pip install -r requirements.txt
```

##### 训练参数配置

```
config.py  

超参数、albert模型 配置文件
```

##### 数据准备

```
参考 data/README.md 文件 
```

##### 训练模型

```
python main.py

```

##### 查看训练过程日志记录(tensorboard)

```
tensorboard.exe --logdir=logs
```




