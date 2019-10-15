import torch.utils.data as data
import numpy as np
import torch
from loaddataset import DataSet, DataSetEval
from torchvision import transforms

class SignalData(data.Dataset):

    def __init__(self, train=True, transform=None, target_transform=None):
        path_dataset = './data/RML2016.10a_dict.pkl'
        dataset = DataSet(path_dataset)
        X_train, Y_train, X_test, Y_test, classes = dataset.getTrainAndTest()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.classes = classes
        if self.train:
            self.train_data, self.train_labels = torch.from_numpy(np.array(X_train).reshape(-1, 16, 16)), \
                                                 torch.from_numpy(np.array(Y_train))
        else:
            self.test_data, self.test_labels = torch.from_numpy(np.array(X_test).reshape(-1, 16, 16)), \
                                               torch.from_numpy(np.array(Y_test))

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class SignalDataEval(data.Dataset):

    def __init__(self, train=True, transform=None, target_transform=None):
        path_dataset = './data/RML2016.10a_dict.pkl'
        dataset = DataSetEval(path_dataset)
        X_train, Y_train, X_test, Y_test, classes = dataset.getTrainAndTest()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.classes = classes
        if self.train:
            self.train_data, self.train_labels = torch.from_numpy(np.array(X_train).reshape(-1, 16, 16)), \
                                                 torch.from_numpy(np.array(Y_train))
        else:
            self.test_data, self.test_labels = torch.from_numpy(np.array(X_test).reshape(-1, 16, 16)), \
                                               torch.from_numpy(np.array(Y_test))

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)