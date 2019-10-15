import numpy as np
import pickle
import random


class DataSet(object):
    def __init__(self, path, unit_num=1000, rate=0.9, seed=2019, snrs=None):
        self.path = path
        self.unit_num = unit_num
        self.rate = rate
        self.seed = seed

        self.loadDataSet(path, self.unit_num, snrs)
        self.splitData(rate, seed)

    # public methods
    def getX(self):
        return self.X, self.lbl, self.snrs, self.mods
    def getTrainAndTest(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test, self.mods
    def getTrainIndex(self):
        return self.train_idx
    def getTestIndex(self):
        return self.test_idx

    def loadDataSet(self, path, unit_num, snrs):
        # 1. load dataset
        Xd = pickle.load(open(path, 'rb'), encoding='iso-8859-1')
        # print(Xd.keys())
        # print(len(set(map(lambda x: x[0], Xd.keys()))))

        # snrs(20) = -20 -> 18  mods(11) = ['8PSK', 'AM-DSB', ...]
        self.snrs, self.mods = map(
            lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])

        self.X = []
        self.lbl = []

        if snrs == None:
            for mod in self.mods:
                for snr in self.snrs:
                    # 1. all samples
                    # X.append(Xd[(mod, snr)])
                    # for i in range(Xd[(mod, snr)].shape[0]):
                    #     lbl.append((mod, snr))

                    # 2. only unit_num samples
                    self.X.append(Xd[(mod, snr)][0:(unit_num)])
                    for i in range(unit_num):
                        self.lbl.append((mod, snr))
        else:
            for mod in self.mods:
                for snr in self.snrs:
                    # 1. all samples
                    # X.append(Xd[(mod, snr)])
                    # for i in range(Xd[(mod, snr)].shape[0]):
                    #     lbl.append((mod, snr))

                    # 2. only unit_num samples
                    if snr in snrs:
                        self.X.append(Xd[(mod, snr)][0:(unit_num)])
                        for i in range(unit_num):
                            self.lbl.append((mod, snr))

        self.X = np.vstack(self.X)


    def to_onehot(self, yy):
        yy1 = np.zeros([len(yy), max(yy) + 1])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1


    def splitData(self, rate = 0.9, seed=2019):
        np.random.seed(seed)
        n_examples = self.X.shape[0]
        n_train = int(n_examples * rate)

        self.train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
        self.test_idx = list(set(range(0, n_examples)) - set(self.train_idx))
        #random.shuffle(self.test_idx)
        self.X_train = self.X[self.train_idx]
        self.X_test = self.X[self.test_idx]

        self.Y_train = list(map(lambda x: self.mods.index(self.lbl[x][0]), self.train_idx))
        self.Y_test = list(map(lambda x: self.mods.index(self.lbl[x][0]), self.test_idx))

        classes = self.mods
        # key = []
        #
        # for label in range(len(classes)):
        #     for i in np.where(np.array(self.Y_train) == label)[0]:
        #         if label not in [1, 2]:
        #             key.append(i)

        # self.X_train = self.X_train[key]
        # self.Y_train = np.array(self.Y_train)[key]

        # type(X_train.shape[1:])
        # in_shp = list(X_train.shape[1:])
        # print(X_train.shape, in_shp)

        return self.X_train, self.Y_train, self.X_test, self.Y_test, classes


class DataSetEval(object):
    def __init__(self, path, unit_num=1000, rate=0.9, seed=2019, snrs=None):
        self.path = path
        self.unit_num = unit_num
        self.rate = rate
        self.seed = seed

        self.loadDataSet(path, self.unit_num, snrs)
        self.splitData(rate, seed)

    # public methods
    def getX(self):
        return self.X, self.lbl, self.snrs, self.mods

    def getTrainAndTest(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test, self.mods

    def getTrainIndex(self):
        return self.train_idx

    def getTestIndex(self):
        return self.test_idx

    def loadDataSet(self, path, unit_num, snrs):
        # 1. load dataset
        Xd = pickle.load(open(path, 'rb'), encoding='iso-8859-1')
        # print(Xd.keys())
        # print(len(set(map(lambda x: x[0], Xd.keys()))))

        # snrs(20) = -20 -> 18  mods(11) = ['8PSK', 'AM-DSB', ...]
        self.snrs, self.mods = map(
            lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])

        self.X = []
        self.lbl = []

        if snrs == None:
            for mod in self.mods:
                for snr in self.snrs:
                    # 1. all samples
                    # X.append(Xd[(mod, snr)])
                    # for i in range(Xd[(mod, snr)].shape[0]):
                    #     lbl.append((mod, snr))

                    # 2. only unit_num samples
                    self.X.append(Xd[(mod, snr)][0:(unit_num)])
                    for i in range(unit_num):
                        self.lbl.append((mod, snr))
        else:
            for mod in self.mods:
                for snr in self.snrs:
                    # 1. all samples
                    # X.append(Xd[(mod, snr)])
                    # for i in range(Xd[(mod, snr)].shape[0]):
                    #     lbl.append((mod, snr))

                    # 2. only unit_num samples
                    if snr in snrs:
                        self.X.append(Xd[(mod, snr)][0:(unit_num)])
                        for i in range(unit_num):
                            self.lbl.append((mod, snr))

        self.X = np.vstack(self.X)

    def to_onehot(self, yy):
        yy1 = np.zeros([len(yy), max(yy) + 1])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1


    def splitData(self, rate = 0.9, seed=2019):
        np.random.seed(seed)
        n_examples = self.X.shape[0]
        n_train = int(n_examples * rate)

        self.train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
        self.test_idx = list(set(range(0, n_examples)) - set(self.train_idx))
        #random.shuffle(self.test_idx)
        self.X_train = self.X[self.train_idx]
        self.X_test = self.X[self.test_idx]

        self.Y_train = list(map(lambda x: self.mods.index(self.lbl[x][0]), self.train_idx))
        self.Y_test = list(map(lambda x: self.mods.index(self.lbl[x][0]), self.test_idx))

        classes = self.mods
        key_train = []
        key_eval = []

        for label in range(len(classes)):
            for i in np.where(np.array(self.Y_train) == label)[0]:
                if label in [1, 2]:
                    key_eval.append(i)
                else:
                    key_train.append(i)

        # type(X_train.shape[1:])
        # in_shp = list(X_train.shape[1:])
        # print(X_train.shape, in_shp)

        self.X_eval = self.X_train[key_eval]
        self.Y_eval = np.array(self.Y_train)[key_eval]

        self.X_train = self.X_train[key_train]
        self.Y_train = np.array(self.Y_train)[key_train]

        return self.X_train, self.Y_train, self.X_test, self.X_test, classes


path_dataset = './data/RML2016.10a_dict.pkl'
dataset = DataSet(path_dataset)
# X(220000 * (2, 128)) and lbl(220000 * 1) is whole dataset
# snrs(20) = -20 -> 18, mods(11) = ['8PSK', 'AM-DSB', ...]
X, lbl, snrs, mods = dataset.getX()
# X_train(176000) Y_train(176000 * 11) classes(11)=mods
X_train, Y_train, X_test, Y_test, classes = dataset.getTrainAndTest()
in_shp = list(X_train.shape[1:]) # (2, 128)

for i in range(X_train.shape[0]):
    print(X_train[i][np.where(X_train[i]==np.max(X_train[i]))])
    X_train[i][np.where(X_train[i]>=0.0)]/=np.max(X_train[i])
    X_train[i][np.where(X_train[i]<0.0)]/=-np.min(X_train[i])
    break