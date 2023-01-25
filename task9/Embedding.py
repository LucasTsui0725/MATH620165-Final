import torch.nn as nn
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

class GenerateEmbedding():
    def __init__(self, embedding_config):
        self._embedding_config = embedding_config
        self.method = self._embedding_config['embedding_method']
        self._check_config()
    
    def _check_config(self):
        default_config = {
            'embedding_method': 'random-embedding', 
            'embedding_dim': 128, 
            'padding_idx': 0
        }
        if 'num_embeddings' not in self._embedding_config:
            raise ValueError('Num_embeddings not in JobConfig!')
        
        if self.method not in ['random-embedding', 'pretrained-embedding']:
            raise ValueError("Embedding method error!")
        
        if 'padding_idx' not in self._embedding_config:
            self._embedding_config['padding_idx'] = default_config['padding_idx']

    def __call__(self):
        if self.method == 'random-embedding':
            embedding = self._random_embedding()
        elif self.method == 'pretrained-embedding':
            embedding = self._pretrained_embedding()
        return embedding
    
    def _random_embedding(self):
        print('Generate random word embedding...')
        num_embeddings = self._embedding_config['num_embeddings']
        embedding_dim = self._embedding_config['embedding_dim']
        padding_idx = self._embedding_config['padding_idx']
        return nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=padding_idx)

    def _pretrained_embedding(self):
        if 'pretrained_vocab' not in self._embedding_config:
            raise ValueError("Can't find pretrained vocab!")
        print("Loading pretraining vector...")
        vocab = self._embedding_config['pretrained_vocab'] # torchtext.legacy.vocab.Vocab
        num_embeddings = self._embedding_config['num_embeddings']
        embedding_dim = self._embedding_config['embedding_dim']
        padding_idx = self._embedding_config['padding_idx']
        embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        embedding_layer.weight.data.copy_(vocab.vectors)
        return embedding_layer
