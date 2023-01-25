import os
import torch
import numpy as np
from datetime import datetime

from utils import count_parameters
from init_parameters import xavier_normal_initialization, xavier_uniform_initialization

def calc_acc(predictions, labels):
    top_predictions = predictions.argmax(1, keepdim = True)
    correct = top_predictions.eq(labels.view_as(top_predictions)).sum()
    accuracy = correct.float() / labels.shape[0]
    return accuracy

def train(model, iterator, optimizer, loss_function, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for _, batch in enumerate(iterator):
        batch_size = batch.batch_size
        text = batch.Content.T.to(device)
        label = batch.Label.to(device).reshape(batch_size, )

        optimizer.zero_grad()
    
        prediction = model(text)
        loss = loss_function(prediction, label)
        acc = calc_acc(prediction, label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, loss_function, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            batch_size = batch.batch_size
            text = batch.Content.T.to(device)
            label = batch.Label.to(device).reshape(batch_size, )

            prediction = model(text)
            loss = loss_function(prediction, label)
            acc = calc_acc(prediction, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience."""
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

class Trainer():
    def __init__(self, jobConfig):
        self._jobConfig = jobConfig
        self._check_config()
        self._initialize_method = self._jobConfig['initialize_method']
        self._num_epochs = self._jobConfig['num_epochs']
        self._lr = self._jobConfig['lr']
        self._early_stopping = self._jobConfig['early_stopping']
        self._patience = self._jobConfig['patience']
        self._device = self._jobConfig['device']
        self._save_path = self._jobConfig['save_path']

        self._train_loss_list = []
        self._test_loss_list = []
        self._train_acc_list = []
        self._test_acc_list = []

    def _check_config(self):
        default_config = {
            'initialize_method': 'normal', 
            'num_epochs': 10000, 
            'lr': 0.01, 
            'early_stopping': True, 
            'patience': 20, 
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
            'save_path': None
        }
        input_parameters = set(self._jobConfig.keys())
        default_parameters = set(default_config.keys())
        add_parameters = default_parameters.difference(input_parameters)

        for para in add_parameters:
            self._jobConfig[para] = default_config[para]
    
    def _clear_cache(self):
        self._train_acc_list.clear()
        self._test_acc_list.clear()
        self._train_loss_list.clear()
        self._test_loss_list.clear()

    def __call__(self, model=None, train_iterator=None, valid_iterator=None, optimizer=None, loss_function=None):
        kwargs = locals()
        kwargs.pop('self')
        for each_para in kwargs:
            if kwargs[each_para] is None:
                raise ValueError('Parameter: \'{}\' is missing!'.format(each_para))
        self._clear_cache()

        task_create_time = datetime.now().strftime("%b_%d_%Y_%H-%M-%S") 
        if self._save_path is None:
            self._save_path = os.getcwd() + '/log/' + task_create_time
            if not os.path.exists(self._save_path):
                os.makedirs(self._save_path)
            
        print('Loading Model...')
        parameters_num = count_parameters(model)
        print('The model has {} parameters.'.format(parameters_num))

        print("Parameters Initializing...")
        if self._initialize_method == 'normal':
            model.apply(xavier_normal_initialization)
        elif self._initialize_method == 'uniform':
            model.apply(xavier_uniform_initialization)
        model = model.to(self._device)
        loss_function = loss_function.to(self._device)

        early_stopping = EarlyStopping(patience=self._patience, verbose=True)

        training_start_time = datetime.now()
        for epoch in range(self._num_epochs):
            train_loss, train_acc = train(model=model, 
                                          iterator=train_iterator, 
                                          optimizer=optimizer, 
                                          loss_function=loss_function, 
                                          device=self._device)
            
            valid_loss, valid_acc = evaluate(model=model, 
                                             iterator=valid_iterator, 
                                             loss_function=loss_function, 
                                             device=self._device)

            self._train_acc_list.append(train_acc)
            self._test_acc_list.append(valid_acc)
            self._train_loss_list.append(train_loss)
            self._test_loss_list.append(valid_loss)

            early_stopping(valid_loss, model, model_path=self._save_path+"/model.pth")
            if early_stopping.early_stop:
                print("\033[31mEarly stopping\033[0m")
                break

            if epoch % 20 == 0:
                print("Time:{}  Epoch:{}\n Train_Loss:{}, Valid_Loss:{}\n Train_acc:{}, Valid_acc:{}\n".format(datetime.now().strftime("%b_%d_%Y_%H-%M-%S"), epoch, train_loss, valid_loss, train_acc, valid_acc))
                print("********************************************************************")
                with open(self._save_path + '/training_log.txt', 'a') as f:
                    f.write("Time:{}  Epoch:{}\n Train_Loss:{}, Valid_Loss:{}\n Train_acc:{}, Valid_acc:{}\n".format(datetime.now().strftime("%b_%d_%Y_%H-%M-%S"), epoch, train_loss, valid_loss, train_acc, valid_acc))
                    f.write("********************************************************************\n")
        training_end_time = datetime.now()
        total_training_time = (training_end_time - training_start_time).total_seconds()

        with open(self._save_path + '/training_log.txt', 'a') as f:
            f.write("\n\n")
            f.write("Train Loss: {}\n".format(str(self._train_loss_list)))
            f.write("Test Loss: {}\n".format(str(self._test_loss_list)))
            f.write("Train Accuracy: {}\n".format(str(self._train_acc_list)))
            f.write("Test Accuracy: {}\n".format(str(self._test_acc_list)))
            f.write("Training Time: {}(s)\n".format(total_training_time))