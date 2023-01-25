import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from Embedding import GenerateEmbedding

class CNN(nn.Module):
    def __init__(self, jobConfig):
        super(CNN, self).__init__()
        self.jobConfig = jobConfig
        self._embedding_config = self.jobConfig['embedding_config']
        self._model_config = self.jobConfig['model_config']
        self._training_config = self.jobConfig['training_config']
        self._check_config()

        self._embedding_method = self._embedding_config['embedding_method']
        self._embedding_dim = self._embedding_config['embedding_dim']
        self._padding_idx = self._embedding_config['padding_idx']
        self._kernel_size = self._model_config['kernel_size']
        self._feature_dim = self._model_config['feature_dim']
        self._dropout_ratio = self._model_config['dropout_ratio']
        self._max_sentence_length = self._model_config['max_sentence_length']
        self._num_class = self._model_config['num_class']
        
        self.embedding = GenerateEmbedding(self._embedding_config)()
        # nn.Conv1d()参数：（一维卷积，用于文本数据，只对宽度进行卷积，对高度不卷积）
        #   in_channels -- 词向量维度
        #   out_channels -- 卷积产生的通道数，即将词向量的维度从in_channels变为out_channels
        #   kernel_size -- 卷积核的尺寸，卷积核的第二个维度由in_channels决定，实际卷积核大小为kernel_size * in_channels
        #   padding -- padding方法
        # 
        # nn.MaxPool1d()参数：
        #   kernel_size -- pooling的窗口大小
        #   stride -- pooling的窗口移动步长
        #   padding -- padding方法
        #   
        self.conv_layers = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('conv_' + str(k), nn.Conv1d(in_channels=self._embedding_dim, out_channels=self._feature_dim, kernel_size=k)),
                ('ReLu', nn.ReLU()),
                ('MaxPool', nn.MaxPool1d(kernel_size=self._max_sentence_length - k + 1))
            ])) for k in self._kernel_size])
        self.fc = nn.Linear(in_features=len(self._kernel_size) * self._feature_dim, 
                            out_features= self._num_class)

    def _check_config(self):
        default_embedding_config = {
            'embedding_method': 'random-embedding', 
            'embedding_dim': 128, 
            'padding_idx': 0
        }
        
        input_embedding_parameters = set(self._embedding_config.keys())
        default_embedding_paremeters = set(default_embedding_config.keys())
        add_parameters = default_embedding_paremeters.difference(input_embedding_parameters)

        for para in add_parameters:
            self._embedding_config[para] = default_embedding_config[para]
        
        if isinstance(self._model_config['kernel_size'], int):
            self._model_config['kernel_size'] = [self._model_config['kernel_size']]
    
    def forward(self, input_text):
        # input_text: [batch_size, sentence_length]
        text_embedding = self.embedding(input_text)
        text_embedding = text_embedding.permute(0, 2, 1)
        # text_embedding: [batch_size, embedding_dim, sentence_length]

        conved = [conv_layer(text_embedding) for conv_layer in self.conv_layers]
        conved = torch.cat(conved, dim=1)
        conved = conved.squeeze(2)
        conved = F.dropout(conved, self._dropout_ratio)
        conved = self.fc(conved)
        return conved
