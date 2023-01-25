import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter, OrderedDict

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# nltk.download('stopwords')
porter = PorterStemmer()
stop_words = stopwords.words('english')

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab


def get_content(file_path):
    '''获取当前文件下的全部文本内容'''
    content = ''
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip('\n')
            line = line.strip('\r')
            content += line
    return content

def get_all_files(root):
    '''获取当前目录下全部文本'''
    content_list = []
    file_list = os.listdir(root)
    # file_iter = tqdm(enumerate(file_list), total=len(file_list), leave=False)
    for file in tqdm(file_list, leave=False):
        file_path = os.path.join(root, file)
        if os.path.isfile(file_path):
            content = get_content(file_path)
            content = preprocess(content)
            content_list.append(content)
    return content_list

def preprocess(input_sentence):
    input_sentence = re.sub("[^a-zA-Z0-9]", " ", input_sentence)
    word_list = [word.lower() for word in input_sentence.split() if word.lower() not in stop_words]
    # word_list = [porter.stem(word) for word in word_list]
    return " ".join(word_list)

class EarlyStopping():
    def __init__(self, patience=20, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    
    def __call__(self, val_loss, model, model_path=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_path=None):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print("Validation loss decrease ({:.6f} --> {:.6f}).".format(self.val_loss_min, val_loss))
        if model_path is None:
            torch.save(model.state_dict(), 'checkpoint.pth')
        elif isinstance(model_path, str):
            torch.save(model.state_dict(), model_path)
        self.val_loss_min = val_loss

class SpecialTokens:
    PAD = '<|pad|>'
    UNK = '<|unk|>'
    SOS = '<|start-of-sentence|>'
    EOS = '<|end-of-sentence|>'

class Tokenizer():
    def __init__(self, tokenizer='basic_english', max_length=None):
        self.tokenizer = get_tokenizer(tokenizer)
        self.max_length = max_length
        self._check_config()

    def _check_config(self):
        if self.max_length is not None:
            if not isinstance(self.max_length, int):
                raise ValueError("Max-length should be None or an Integer!")

    def __call__(self, input_sentence):
        tokens = self.tokenizer(input_sentence)
        tokens = [word for word in tokens if len(word) > 1]
        if self.max_length is not None:
            tokens = tokens[: self.max_length]
        return tokens

def build_vocab(input_data, tokenizer, use_special_token=True, **kwargs):
    print("Building Vocabulary...")
    token_frequence = Counter()
    for sentence in input_data:
        tokens = tokenizer(sentence)
        token_frequence.update(tokens)
    sorted_by_freq_tuples = sorted(token_frequence.items(), key=lambda x: x[1], reverse=True)
    if use_special_token:
        max_freq = sorted_by_freq_tuples[0][1]
        sorted_by_freq_tuples.insert(0, (SpecialTokens.EOS, max_freq + 1))
        sorted_by_freq_tuples.insert(0, (SpecialTokens.SOS, max_freq + 2))
        sorted_by_freq_tuples.insert(0, (SpecialTokens.UNK, max_freq + 3))
        sorted_by_freq_tuples.insert(0, (SpecialTokens.PAD, max_freq + 4))
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    vocabulary = vocab(ordered_dict, **kwargs)
    return vocabulary

def count_parameters(model):
    r"""
        Calculate parametes that need to update in a pytorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
