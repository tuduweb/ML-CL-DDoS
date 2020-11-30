import argparse
import globalvar as gl    #添加全局变量管理模块

def test_for_sys(year, name, body):
    print('the year is', year)
    print('the name is', name)
    print('the body is', body)

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--cuda', '-c', help='是否应用cuda', default=0)
parser.add_argument('--local', '-l', help='是否在本地运行', default=0)
parser.add_argument('--test', '-t', help='测试模型', default=0)
parser.add_argument('--model_batch_test', '-bt', help='键入测试模型ID', default=0)
parser.add_argument('--model_batch_save', '-bs', help='GPU模型转换为CPU', default=0)
#parser.add_argument('--body', '-b', help='body 属性，必要参数', required=False)
global_args = parser.parse_args()

import copy
import pickle
import pandas as pd

from datetime import datetime
import os
import shutil
import unittest
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F

from context import FederatedAveragingGrads
from context import PytorchModel
from learning_model import FLModel
from preprocess import get_test_loader
from preprocess import UserRoundData
from train import user_round_train


class ParameterServer(object):

    def __init__(self, init_model_path=None, testworkdir='', model_arg_obj=None, model_obj=None,learn_rate=0.001):
        self.round = 0
        self.rounds_info = {}
        self.rounds_model_path = {}
        self.current_round_grads = []
        self.init_model_path = init_model_path
        self.aggr = FederatedAveragingGrads(
            model=PytorchModel(torch=torch,
                               model_class=FLModel,
                               init_model_path=self.init_model_path,
                               optim_name='Adam',
                               cuda=gl.get_value("use_cuda"),
                               lr=learn_rate
                               ),
            framework='pytorch',
        )

        self.rounds_model_arg_objs = {}
        self.model_arg_obj = model_arg_obj
        self.model_obj = model_obj
        self.rounds_model_objs = {}
        self.last_model_obj = None

        self.testworkdir = testworkdir
        self.outputdir = ''
        self.round_savemodel_int = gl.get_value("round_savemodel_int")

        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)

    def get_latest_model_path(self):
        if not self.rounds_model_path:
            return self.init_model_path

        if self.round in self.rounds_model_path:
            return self.rounds_model_path[self.round]

        return self.rounds_model_path[self.round - 1]

    def get_latest_model_obj(self):
        if not self.last_model_obj:
            return self.model_obj

        return self.last_model_obj

    def get_latest_model_arg_obj(self):
        if not self.rounds_model_arg_objs:
            return self.model_arg_obj

        if self.round in self.rounds_model_arg_objs:
            return self.rounds_model_arg_objs[self.round]

        return self.rounds_model_arg_objs[self.round - 1]



    def receive_grads_info(self, grads):
        self.current_round_grads.append(grads)

    def save_model(self):
        path = os.path.join(self.outputdir,
                            'round-{round}-model.pkl'.format(round=self.round))
        info = self.aggr.save_model(path=path)  # ps.aggr.save_model

    # aggregate 总数; 合计;
    def aggregate(self):
        #联邦学习操作 输入的是一个list
        self.aggr(self.current_round_grads)

        if gl.get_value("is_local") == False:
            # 文件操作
            path = os.path.join(self.testworkdir,
                                'round-{round}-model.pkl'.format(round=self.round))
            self.rounds_model_path[self.round] = path
            if (self.round - 1) in self.rounds_model_path:
                if os.path.exists(self.rounds_model_path[self.round - 1]):
                    os.remove(self.rounds_model_path[self.round - 1])

            # 保存模型
            info = self.aggr.save_model(path=path) #ps.aggr.save_model
        else:
            #self.rounds_model_arg_objs[self.round] = self.aggr.model.model.state_dict() #PytorchModel 需要这个class下的model.state_dict()
            self.model_obj = self.aggr.model.model #直接把model赋值过去 注意要是cpu类型
            info = None

        self.round += 1
        self.current_round_grads = []

        if (self.round % self.round_savemodel_int == 1):
            self.save_model()

        return info

from collections import Counter

class FedAveragingGradsTestSuit(unittest.TestCase):
    RESULT_DIR = 'result'
    N_VALIDATION = 10000
    TEST_BASE_DIR = '/tmp/'

    def setUp(self):
        if gl.get_value("use_cuda"):
            #self.TEST_BASE_DIR = 'C:/tmp/'
            print("Hello User: using cuda~")
            self.use_cuda = True
        else:
            self.use_cuda = False

        self.seed = 0
        #self.use_cuda = True
        self.batch_size = 256
        self.test_batch_size = 8192
        self.lr = 0.001 #学习率,上传的程序没有修改成功
        self.n_max_rounds = 10000
        self.log_interval = 10
        self.n_round_samples = 1600 #随机抽取的样本数 #df:1600
        self.testbase = self.TEST_BASE_DIR
        self.testworkdir = os.path.join(self.testbase, 'competetion-test')
        self.testIntRound = 100 #测试间隔
        self.savemodel_interval = 100 #保存模型间隔

        self.now_round = 0
        self.outputdir = os.path.join(self.RESULT_DIR, str(gl.get_value('start_time')))

        gl.set_value("round_savemodel_int", self.savemodel_interval)

        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)

        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

        self.init_model_path = os.path.join(self.testworkdir, 'init_model.pkl')
        torch.manual_seed(self.seed)

        # 如果没有初始化模型,那么执行:模型初始化
        if not os.path.exists(self.init_model_path):
            torch.save(FLModel().state_dict(), self.init_model_path) # model.state_dict()其实返回的是一个OrderDict，存储了网络结构的名字和对应的参数

        if gl.get_value("is_local"):
            self.ps = ParameterServer(init_model_path=self.init_model_path, testworkdir=self.testworkdir,
                                      model_obj=FLModel(), learn_rate=self.lr)
        else:
            self.ps = ParameterServer(init_model_path=self.init_model_path, testworkdir=self.testworkdir, learn_rate=self.lr)

        self.ps.outputdir = self.outputdir

        if not os.path.exists(self.RESULT_DIR):
            os.makedirs(self.RESULT_DIR)

        self.urd = UserRoundData()
        self.n_users = self.urd.n_users

    def _clear(self):
        shutil.rmtree(self.testworkdir)

    def tearDown(self):
        self._clear()

    def test_federated_averaging(self):
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if self.use_cuda else "cpu")

        training_start = datetime.now()
        model = None

        if gl.get_value("is_local"):
            for r in range(1, self.n_max_rounds + 1):
                # 从参数服务器中来
                model_obj = self.ps.get_latest_model_obj()

                start = datetime.now()
                for u in range(0, self.n_users):# 每个节点
                    model = copy.deepcopy(model_obj) #拷贝,在CPU中
                    model = model.to(device) #放到GPU/CPU中
                    x, y = self.urd.round_data(
                        user_idx=u,
                        n_round=r,
                        n_round_samples=self.n_round_samples)
                    #print(Counter(y))
                    grads = user_round_train(X=x, Y=y, model=model, device=device, batch_size= self.batch_size, debug=False)
                    # 参数服务器接受梯度.没有带节点信息.函数内以append方式接收
                    self.ps.receive_grads_info(grads=grads)

                # 执行联邦学习
                self.ps.aggregate()
                print('\nRound {} cost: {}, total training cost: {}'.format(
                    r,
                    datetime.now() - start,
                    datetime.now() - training_start,
                ))

                if model is not None and r % self.testIntRound == 0:
                    self.predict(model,
                                 device,
                                 self.urd.uniform_random_loader(self.N_VALIDATION),
                                 prefix="Train")  # 用的train数据
                    # 预测 无标签的数据,并保存到result.txt
                    self.save_testdata_prediction(model=model, device=device)  # 用的pkl数据

        else:

            for r in range(1, self.n_max_rounds + 1):
                path = self.ps.get_latest_model_path()
                start = datetime.now()
                for u in range(0, self.n_users):
                    model = FLModel()
                    model.load_state_dict(torch.load(path))
                    model = model.to(device)
                    x, y = self.urd.round_data(
                        user_idx=u,
                        n_round=r,
                        n_round_samples=self.n_round_samples)
                    grads = user_round_train(X=x, Y=y, model=model, device=device, batch_size= self.batch_size,  debug=False)
                    # 参数服务器接受梯度.没有带u信息
                    self.ps.receive_grads_info(grads=grads)

                self.ps.aggregate()
                print('\nRound {} cost: {}, total training cost: {}'.format(
                    self.now_round,
                    datetime.now() - start,
                    datetime.now() - training_start,
                ))

                if model is not None and r % self.testIntRound == 0:
                    self.predict(model,
                                 device,
                                 self.urd.uniform_random_loader(self.N_VALIDATION),
                                 prefix="Train")# 用的train数据
                    # 预测 无标签的数据,并保存到result.txt
                    self.save_testdata_prediction(model=model, device=device)#用的pkl数据


        self.ps.save_model() # 保存当前轮数的模型
        if model is not None:
            self.save_testdata_prediction(model=model, device=device)

    def save_prediction(self, predition):
        if isinstance(predition, (np.ndarray, )):
            predition = predition.reshape(-1).tolist()
        self.now_round += 1

        saved_filename = '%d-%d.txt' % (gl.get_value('start_time'), self.now_round)
        with open(os.path.join(self.outputdir, saved_filename), 'w') as fout:
            #fout.writelines(os.linesep.join([str(n) for n in predition])) #根据系统类型添加换行符
            fout.writelines("\n".join([str(n) for n in predition]))
            fout.close()

    def save_testdata_prediction(self, model, device):
        # 测试的时候batch_size 因为没有学习过程 那么越大越好？
        loader = get_test_loader(batch_size=self.test_batch_size)#测试时候的batchsize
        prediction = []
        with torch.no_grad():
            for data in loader:
                pred = model(data.to(device)).argmax(dim=1, keepdim=True)
                prediction.extend(pred.reshape(-1).tolist())

        self.save_prediction(prediction)

    # 使用随机的训练数据来测试当前模型准确率
    def predict(self, model, device, test_loader, prefix=""):
        model.eval()
        test_loss = 0
        correct = 0
        prediction = []
        real = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(
                    output, target,
                    reduction='sum').item()  # sum up batch loss
                pred = output.argmax(
                    dim=1,
                    keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                prediction.extend(pred.reshape(-1).tolist())
                real.extend(target.reshape(-1).tolist())

        test_loss /= len(test_loader.dataset)
        acc = 100. * correct / len(test_loader.dataset)
        print(classification_report(real, prediction))
        print(
            '{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                prefix, test_loss, correct, len(test_loader.dataset), acc), )



from preprocess import CompDataset
import re


class LocalTestModelTestSuit(unittest.TestCase):
    def setUp(self):
        if gl.get_value("use_cuda"):
            #self.TEST_BASE_DIR = 'C:/tmp/'
            print("Hello Tester: using cuda~")
            self.use_cuda = True
        else:
            self.use_cuda = False

        self.test_model_path = gl.get_value("test_model_path")
        self.test_batch_size = 4096
        self.outputdir = 'result'

        self.test_models_path = None


        # if gl.get_value("is_local"):
        #     self.ps = ParameterServer(init_model_path=self.init_model_path, testworkdir=self.testworkdir,
        #                               model_obj=FLModel(), learn_rate=self.lr)
        # else:
        #self.ps = ParameterServer(init_model_path=self.test_model_path, testworkdir='', learn_rate=0.001)


    def ModelLocalTest(self):
        if gl.get_value("test_model_path_id") is not None:
            self.test_models_path = os.path.join('result-autotest', str(gl.get_value("test_model_path_id")), 'model')
            if os.path.exists(self.test_models_path) is None:
                print('Test Models path id err')
                return

        if self.test_model_path is not None and os.path.exists(self.test_model_path) is None:
            print('Test Model cant be found')
            return

        device = torch.device("cuda" if self.use_cuda else "cpu")

        TESTDATA_PATH = '../../dataset/new_test/1606711421-1156737.pkl'

        with open(TESTDATA_PATH, 'rb') as fin:
            data = pickle.load(fin)
            data['X'] = data['X']
            data['X'] = pd.DataFrame(data['X'])
            data['X'] = data['X'].drop(data['X'].columns[[28, 29, 30, 31, 41, 42, 43, 44, 48, 54, 55, 56, 57, 58, 59, 76]], axis=1)


            scaler = MinMaxScaler()
            data['X'] = scaler.fit_transform(data['X'])

        data = CompDataset(X=data['X'], Y=data['Y'])

        test_loader = torch.utils.data.DataLoader(
            data,
            batch_size=self.test_batch_size,
            shuffle=False,
        )
        #gl.set_value("test_model_path_id")

        model = FLModel()

        if(self.test_model_path is not None):
            model.load_state_dict(torch.load(self.test_model_path))

            if gl.get_value('model_convert_to_gpu') is not None:
                model.to(torch.device("cpu"))
                portion = os.path.splitext(self.test_model_path)
                new_model = portion[0] + '.cmodel'
                torch.save(model.state_dict(), new_model)

            model.to(device)
            model.eval()
            prediction = []
            real = []
            correct = 0
            test_loss = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    #pred = model(data.to(device)).argmax(dim=1, keepdim=True)
                    test_loss += F.nll_loss(
                        output, target,
                        reduction='sum').item()  # sum up batch loss
                    pred = output.argmax(
                        dim=1,
                        keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    prediction.extend(pred.reshape(-1).tolist())
                    real.extend(target.reshape(-1).tolist())

            test_loss /= len(test_loader.dataset)
            acc = 100. * correct / len(test_loader.dataset)

            print(classification_report(real, prediction))
            print(
                '{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    "Validation", test_loss, correct, len(test_loader.dataset), acc), )
            #self.save_prediction(prediction)
        else:
            for root, dirs, fnames in os.walk(self.test_models_path):
                new_fnames = [d for d in fnames if '.pkl' in d]
                new_fnames = sorted(new_fnames, key=lambda i: int(re.match(r'round-(\d+)-model.pkl', i)[1]))
                for fname in new_fnames:
                    print('@'*20)
                    print(fname)
                    model.load_state_dict(torch.load(os.path.join(self.test_models_path, fname)))
                    if gl.get_value('model_convert_to_gpu') is not None:
                        model.to(torch.device("cpu"))
                        portion = os.path.splitext(fname)
                        new_model = portion[0] + '.cmodel'
                        torch.save(model.state_dict(), os.path.join(self.test_models_path, new_model))
                    model.to(device)
                    model.eval()
                    prediction = []
                    real = []
                    correct = 0
                    test_loss = 0
                    with torch.no_grad():
                        for data, target in test_loader:
                            data, target = data.to(device), target.to(device)
                            output = model(data)
                            # pred = model(data.to(device)).argmax(dim=1, keepdim=True)
                            test_loss += F.nll_loss(
                                output, target,
                                reduction='sum').item()  # sum up batch loss
                            pred = output.argmax(
                                dim=1,
                                keepdim=True)  # get the index of the max log-probability
                            correct += pred.eq(target.view_as(pred)).sum().item()
                            prediction.extend(pred.reshape(-1).tolist())
                            real.extend(target.reshape(-1).tolist())

                    test_loss /= len(test_loader.dataset)
                    acc = 100. * correct / len(test_loader.dataset)

                    print(classification_report(real, prediction))
                    print(
                        '{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                            "Validation", test_loss, correct, len(test_loader.dataset), acc), )

    def save_prediction(self, predition):
        if isinstance(predition, (np.ndarray, )):
            predition = predition.reshape(-1).tolist()

        saved_filename = 'localmodel_%d.txt' % (gl.get_value('start_time') )
        with open(os.path.join(self.outputdir, saved_filename), 'w') as fout:
            #fout.writelines(os.linesep.join([str(n) for n in predition])) #根据系统类型添加换行符
            fout.writelines("\n".join([str(n) for n in predition]))
            fout.close()

    # def save_testdata_prediction(self, model, device):
    #     # 测试的时候batch_size 因为没有学习过程 那么越大越好？
    #     loader = get_test_loader(batch_size=self.test_batch_size)#测试时候的batchsize
    #     prediction = []
    #     with torch.no_grad():
    #         for data in loader:
    #             pred = model(data.to(device)).argmax(dim=1, keepdim=True)
    #             prediction.extend(pred.reshape(-1).tolist())
    #
    #     self.save_prediction(prediction)

    def _clear(self):
        #shutil.rmtree(self.testworkdir)
        pass
    def tearDown(self):
        self._clear()







def suite():
    suite = unittest.TestSuite()
    if gl.get_value('test_model_path') is not None or gl.get_value('test_model_path_id') is not None:
        suite.addTest(LocalTestModelTestSuit('ModelLocalTest'))
    else:
        suite.addTest(FedAveragingGradsTestSuit('test_federated_averaging'))
    return suite


def main():
    runner = unittest.TextTestRunner()
    runner.run(suite())

import time

import sys
class Logger(object):
  def __init__(self, filename="Default.log"):
    self.terminal = sys.stdout
    self.log = open(filename, "a")
  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)
  def flush(self):
    pass


if __name__ == '__main__':


    try:
        gl._init()
        gl.set_value("start_time", int(time.time()))
        gl.set_value("program_args", global_args)
        gl.set_value("use_cuda", False if global_args.cuda == 0 else True)
        gl.set_value("is_local", False if global_args.local == 0 else True)
        gl.set_value("test_model_path", None if global_args.test == 0 else global_args.test)
        gl.set_value("test_model_path_id", None if global_args.model_batch_test == 0 else int(global_args.model_batch_test))
        gl.set_value("model_convert_to_gpu", None if global_args.model_batch_save == 0 else int(global_args.model_batch_save))
    except Exception as e:
        print(e)

    # 输出保存到文本
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()

    if gl.get_value('test_model_path') is None or gl.get_value('test_model_path_id') is None:
        if os.path.exists("./log/") == False:
            os.makedirs("./log/")
        sys.stdout = Logger("./log/%d.txt" % gl.get_value("start_time"))

    main()
