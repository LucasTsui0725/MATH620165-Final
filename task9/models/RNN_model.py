import torch
import torch.nn as nn
import torch.nn.functional as F
from Embedding import GenerateEmbedding

class RNN(nn.Module):
    def __init__(self, jobConfig):
        super(RNN, self).__init__()
        self.jobConfig = jobConfig
        self._embedding_config = self.jobConfig['embedding_config']
        self._model_config = self.jobConfig['model_config']
        self._training_config = self.jobConfig['training_config']
        self._check_config()

        self._embedding_method = self._embedding_config['embedding_method']
        self._embedding_dim = self._embedding_config['embedding_dim']
        self._padding_idx = self._embedding_config['padding_idx']
        self._rnn_type = self._model_config['rnn_type'].lower()
        self._hidden_size = self._model_config['hidden_size']
        self._bidirectional = self._model_config['bidirectional']
        self._num_class = self._model_config['num_class']
        
        self.embedding = GenerateEmbedding(self._embedding_config)()
        if self._rnn_type == 'rnn':
            self.layer = nn.RNN(input_size=self._embedding_dim, hidden_size=self._hidden_size, batch_first=True, bidirectional=self._bidirectional)
        elif self._rnn_type == 'lstm':
            self.layer = nn.LSTM(input_size=self._embedding_dim, hidden_size=self._hidden_size, batch_first=True, bidirectional=self._bidirectional)

        self.output_layer = nn.Linear(in_features=self._hidden_size * (2 if self._bidirectional else 1), out_features=self._num_class)

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

    def forward(self, input_text):
        # input_text: [batch_size, sentence_length]
        batch_size = input_text.shape[0]
        text_embedding = self.embedding(input_text)
        # text_embedding = [batch_size, sentence_length, embedding_dim]

        # Input: [batch_size, sentence_length, embedding_dim]
        # Output: [batch_size, sentence_length, num_directions * hidden_size]
        # Hidden_state: [num_layers * num_directions, batch_size, hidden_size]
        if self._rnn_type == 'rnn':
            h0 = torch.randn((2 if self._bidirectional else 1), batch_size, self._hidden_size)
            output, hn = self.layer(text_embedding, h0)
        elif self._rnn_type == 'lstm':
            h0, c0 = torch.randn((2 if self._bidirectional else 1), batch_size, self._hidden_size), torch.randn((2 if self._bidirectional else 1), batch_size, self._hidden_size)
            output, (hn, cn) = self.layer(text_embedding, (h0, c0))
        if self._bidirectional:
            hn = torch.cat((hn[0], hn[1]), dim=1)
        labels = self.output_layer(hn)
        return labels


