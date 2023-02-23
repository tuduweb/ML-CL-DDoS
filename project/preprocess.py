import os
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.preprocessing import MinMaxScaler
import globalvar as gl
import sys

TRAINDATA_DIR = '../../../dataset/train/'
TESTDATA_PATH = '../../../dataset/new_test/1606549610-682356.pkl'
# TESTDATA_PATH = '../../../dataset/new_test/testdata-36788.pkl'
# TESTDATA_PATH = '../../../dataset/new_test/153763.pkl'

ATTACK_TYPES = {
    'snmp': 0,
    'portmap': 1,
    'syn': 2,
    'dns': 3,
    'ssdp': 4,
    'webddos': 5,
    'mssql': 6,
    'tftp': 7,
    'ntp': 8,
    'udplag': 9,
    'ldap': 10,
    'netbios': 11,
    'udp': 12,
    'benign': 13,
}


class CompDataset(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        # zip:打包为元组的列表
        self._data = [(x, y) for x, y in zip(X, Y)]

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


def extract_features(data, has_label=True):  #提取特征
     
    data['SimillarHTTP'] = 0.
    if has_label:
        return data.iloc[:, -80:-1]

    return data.iloc[:, -79:]




class UserRoundData(object):
    def __init__(self):
        if gl.get_value("train_dataset_path"):
            self.data_dir = os.path.abspath(os.path.join(gl.get_value("pwd"), gl.get_value("train_dataset_path")))
        else:
            self.data_dir = os.path.abspath(os.path.join(gl.get_value("pwd"), TRAINDATA_DIR))

        self._user_datasets = []
        self.attack_types = ATTACK_TYPES

        if self._check_path("") < 0:
            pass
        self._load_data()

    def _check_path(self, path):
        self._current_path = os.path.join(sys.path[0], path)
        print("rundir: %s" % self._current_path)

        return 0


    def _get_data(self, fpath):
        if not fpath.endswith('csv'):
            return

        print('Load User Data: ', os.path.basename(fpath))
        data = pd.read_csv(fpath, skipinitialspace=True, low_memory=False)
        x = extract_features(data)
        y = np.array([
            self.attack_types[t.split('_')[-1].replace('-', '').lower()]
            for t in data.iloc[:, -1]
        ])
        x[x == np.inf] = 1.
        x[np.isnan(x)] = 0.
        # x = x.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

        # 去除数据集中无需要的列
        x = x.drop(x.columns[[28, 29, 30, 31, 41, 42, 43, 44, 48, 54, 55, 56, 57, 58, 59, 76]], axis=1)

        x = x.to_numpy().astype(np.float32)
        y = y.astype(np.longlong)


        # x = pd.DataFrame(x)
        # x = x.to_numpy().astype(np.float32)

        #standardScaler = StandardScaler()
        #x = standardScaler.fit_transform(x)
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)
        return (
            x,
            y,
        )

    def _load_data(self):
        isSysTest = gl.get_value("sys_test")

        _user_datasets = []
        self._user_datasets = []
        for root, dirs, fnames in os.walk(self.data_dir):
            for fname in fnames:
                # each file is for each user
                # user data can not be shared among users
                data = self._get_data(os.path.join(root, fname))
                if data is not None:
                    _user_datasets.append(data)

                if isSysTest and len(_user_datasets):
                    break

            if isSysTest and len(_user_datasets):
                break

        if len(_user_datasets) == 0:
            print("datasets path [%s] emtpy" % self.data_dir)
            exit(0)

        for x, y in _user_datasets:
            self._user_datasets.append((
                x,
                y,
            ))

        self.n_users = len(_user_datasets)

    def __str__(self):
        return "preprocess"
        
    def datasets(self):
        return self._user_datasets


    def round_data(self, user_idx, n_round, n_round_samples=-1):
        """Generate data for user of user_idx at round n_round.

        Args:
            user_idx: int,  in [0, self.n_users)
            n_round: int, round number
        """
        if n_round_samples == -1:
            return self._user_datasets[user_idx]

        n_samples = len(self._user_datasets[user_idx][1])
        choices = np.random.choice(n_samples, min(n_samples, n_round_samples))

        return self._user_datasets[user_idx][0][choices], self._user_datasets[
            user_idx][1][choices]

    def uniform_random_loader(self, n_samples, batch_size=1000):
        X, Y = [], []
        n_samples_each_user = n_samples // len(self._user_datasets)
        if n_samples_each_user <= 0:
            n_samples_each_user = 1

        for idx in range(len(self._user_datasets)):
            x, y = self.round_data(user_idx=idx,
                                   n_round=0,
                                   n_round_samples=n_samples_each_user)
            X.append(x)
            Y.append(y)

        data = CompDataset(X=np.concatenate(X), Y=np.concatenate(Y))
        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=min(batch_size, n_samples),
            shuffle=True,
        )

        return train_loader

import pandas
def get_test_loader(batch_size=1000):#default size
    testPath = gl.get_value("test_dataset_path", TESTDATA_PATH)

    with open(testPath, 'rb') as fin:
        data =  pandas.read_pickle(fin)
        #print(data)
        if 'X' in data:
            data['X'] = data['X'].astype(np.float32)
            x = pd.DataFrame(data['X'])
        else:
            x = pd.DataFrame(data)
        #standardScaler = StandardScaler()
        #x = standardScaler.fit_transform(x)
        #data = data['X']
        # x = data.iloc[:, -80:-1]
        # x['SimillarHTTP'] = 0.
        # #y = data.iloc[:, -1]
        #
        # x = x.to_numpy().astype(np.float32)
        # #y = y.to_numpy().astype(np.longlong)
        #
        # x[x == np.inf] = 1.
        # x[np.isnan(x)] = 0.


        x = x.drop(x.columns[[28, 29, 30, 31, 41, 42, 43, 44, 48, 54, 55, 56, 57, 58, 59, 76]], axis=1)
        #x = x.to_numpy().astype(np.float32)

        x = MinMaxScaler().fit_transform(x)


    test_loader = torch.utils.data.DataLoader(
        x,
        batch_size=batch_size,
        shuffle=False,
    )

    return test_loader
