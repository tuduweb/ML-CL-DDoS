import argparse
import globalvar as gl    #添加全局变量管理模块

def test_for_sys(year, name, body):
    print('the year is', year)
    print('the name is', name)
    print('the body is', body)

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--cuda', '-c', help='是否应用cuda', default=0)
parser.add_argument('--local', '-l', help='是否在本地运行', default=0)
#parser.add_argument('--body', '-b', help='body 属性，必要参数', required=False)
global_args = parser.parse_args()

import copy



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
    def __init__(self, init_model_path = None, testworkdir = '', model_arg_obj = None, model_obj = None):
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
                               cuda=gl.get_value("use_cuda")
                               ),
            framework='pytorch',
        )

        self.rounds_model_arg_objs = {}
        self.model_arg_obj = model_arg_obj
        self.model_obj = model_obj
        self.rounds_model_objs = {}
        self.last_model_obj = None

        self.testworkdir = testworkdir
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

    # aggregate 总数; 合计;
    def aggregate(self):
        #联邦学习操作 输入的是一个list
        self.aggr(self.current_round_grads)

        if gl.get_value("is_local") == False:
            # 文件操作
            path = os.path.join(self.testworkdir,
                                'round-{round}-model.md'.format(round=self.round))
            self.rounds_model_path[self.round] = path
            if (self.round - 1) in self.rounds_model_path:
                if os.path.exists(self.rounds_model_path[self.round - 1]):
                    os.remove(self.rounds_model_path[self.round - 1])

            # 保存模型
            info = self.aggr.save_model(path=path)
        else:
            #self.rounds_model_arg_objs[self.round] = self.aggr.model.model.state_dict() #PytorchModel 需要这个class下的model.state_dict()
            self.model_obj = self.aggr.model.model #直接把model赋值过去 注意要是cpu类型
            info = None

        self.round += 1
        self.current_round_grads = []

        return info

from collections import Counter

class FedAveragingGradsTestSuit(unittest.TestCase):
    RESULT_DIR = 'result'
    N_VALIDATION = 10000
    TEST_BASE_DIR = '/tmp/'

    def setUp(self):
        if gl.get_value("use_cuda"):
            self.TEST_BASE_DIR = 'C:/tmp/'
            print("Hello User: using cuda~")
            self.use_cuda = True
        else:
            self.use_cuda = False

        self.seed = 0
        #self.use_cuda = True
        self.batch_size = 64
        self.test_batch_size = 1000
        self.lr = 0.1
        self.n_max_rounds = 5200
        self.log_interval = 10
        self.n_round_samples = 1600 #随机抽取的样本数 #df:1600
        self.testbase = self.TEST_BASE_DIR
        self.testworkdir = os.path.join(self.testbase, 'competetion-test')
        self.testIntRound = 20

        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)

        self.init_model_path = os.path.join(self.testworkdir, 'init_model.md')
        torch.manual_seed(self.seed)

        # 如果没有初始化模型,那么执行:模型初始化
        if not os.path.exists(self.init_model_path):
            torch.save(FLModel().state_dict(), self.init_model_path) # model.state_dict()其实返回的是一个OrderDict，存储了网络结构的名字和对应的参数

        if gl.get_value("is_local"):
            self.ps = ParameterServer(
                                      init_model_path=self.init_model_path,
                                      testworkdir=self.testworkdir,
                                      #model_arg_obj=FLModel().state_dict()
                                      model_obj=FLModel()
                                      )
        else:
            self.ps = ParameterServer(init_model_path=self.init_model_path,
                                  testworkdir=self.testworkdir
                                  )

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
                    print(Counter(y))
                    grads = user_round_train(X=x, Y=y, model=model, device=device)
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

            # for r in range(1, self.n_max_rounds + 1):
            #     # 从参数服务器中来
            #     model_dict_obj = self.ps.get_latest_model_arg_obj()
            #
            #     start = datetime.now()
            #     for u in range(0, self.n_users):# 每个节点
            #         model = FLModel()
            #         model.load_state_dict(model_dict_obj)
            #         model = model.to(device)
            #         x, y = self.urd.round_data(
            #             user_idx=u,
            #             n_round=r,
            #             n_round_samples=self.n_round_samples)
            #         grads = user_round_train(X=x, Y=y, model=model, device=device)
            #         # 参数服务器接受梯度.没有带节点信息.函数内以append方式接收
            #         self.ps.receive_grads_info(grads=grads)
            #
            #     # 执行联邦学习
            #     self.ps.aggregate()
            #     print('\nRound {} cost: {}, total training cost: {}'.format(
            #         r,
            #         datetime.now() - start,
            #         datetime.now() - training_start,
            #     ))
            #
            #     if model is not None and r % self.testIntRound == 0:
            #         self.predict(model,
            #                      device,
            #                      self.urd.uniform_random_loader(self.N_VALIDATION),
            #                      prefix="Train")  # 用的train数据
            #         # 预测 无标签的数据,并保存到result.txt
            #         self.save_testdata_prediction(model=model, device=device)  # 用的pkl数据

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
                    grads = user_round_train(X=x, Y=y, model=model, device=device)
                    # 参数服务器接受梯度.没有带u信息
                    self.ps.receive_grads_info(grads=grads)

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
                                 prefix="Train")# 用的train数据
                    # 预测 无标签的数据,并保存到result.txt
                    self.save_testdata_prediction(model=model, device=device)#用的pkl数据

        if model is not None:
            self.save_testdata_prediction(model=model, device=device)

    def save_prediction(self, predition):
        if isinstance(predition, (np.ndarray, )):
            predition = predition.reshape(-1).tolist()

        with open(os.path.join(self.RESULT_DIR, 'result.txt'), 'w') as fout:
            fout.writelines(os.linesep.join([str(n) for n in predition]))

    def save_testdata_prediction(self, model, device):
        # 测试的时候batch_size 因为没有学习过程 那么越大越好？
        loader = get_test_loader(batch_size=1000)
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


def suite():
    suite = unittest.TestSuite()
    suite.addTest(FedAveragingGradsTestSuit('test_federated_averaging'))
    return suite


def main():
    runner = unittest.TextTestRunner()
    runner.run(suite())

if __name__ == '__main__':
    try:
        gl._init()
        gl.set_value("program_args", global_args)
        gl.set_value("use_cuda", False if global_args.cuda == 0 else True)
        gl.set_value("is_local", False if global_args.local == 0 else True)

    except Exception as e:
        print(e)
    main()
