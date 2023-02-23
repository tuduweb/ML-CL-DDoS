import unittest
import paramunittest

import globalvar as gl    #添加全局变量管理模块


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

    def __init__(self, init_model_path=None, testworkdir='', model_arg_obj=None, model_obj=None,learn_rate=0.001,output_model_dir = '', opt_schedule=None):
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
                               lr=learn_rate,
                               opt_schedule=opt_schedule
                               ),
            framework='pytorch',
        )

        self.rounds_model_arg_objs = {}
        self.model_arg_obj = model_arg_obj
        self.model_obj = model_obj
        self.rounds_model_objs = {}
        self.last_model_obj = None

        self.testworkdir = testworkdir
        self.outputdir = output_model_dir
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


user_datasets = None

@paramunittest.parametrized(
    {"max_rounds": 2500, "n_round_samples": 1600, "batch_size": 320, "init_lr": 0.001,
     "opt_schedule_func": None},

    # {"max_rounds": 5000, "n_round_samples": 1600, "batch_size": 320, "init_lr": 0.005,
    #  "opt_schedule_func": lambda opt: torch.optim.lr_scheduler.MultiStepLR(opt, [600, 2100], 0.2, -1)},


    # {"max_rounds": 3000, "n_round_samples": 2048, "batch_size": 512, "init_lr": 0.001,
    #  "opt_schedule_func": None},
    #
    # {"max_rounds": 4000, "n_round_samples": 2048, "batch_size": 1024, "init_lr": 0.001,
    #  "opt_schedule_func": None},
    #
    # {"max_rounds": 6000, "n_round_samples": 2048, "batch_size": 2048, "init_lr": 0.001,
    #  "opt_schedule_func": None},
    #
    # {"max_rounds": 2000, "n_round_samples": 2048, "batch_size": 512, "init_lr": 0.002,
    #  "opt_schedule_func": None},
    #
    # {"max_rounds": 2000, "n_round_samples": 2048, "batch_size": 512, "init_lr": 0.004,
    #  "opt_schedule_func": None},
    #
    # {"max_rounds": 2000, "n_round_samples": 2048, "batch_size": 512, "init_lr": 0.006,
    #  "opt_schedule_func": None},
    #
    # {"max_rounds": 3000, "n_round_samples": 4096, "batch_size": 512, "init_lr": 0.001,
    #  "opt_schedule_func": None},
    #
    # {"max_rounds": 3000, "n_round_samples": 8192, "batch_size": 512, "init_lr": 0.001,
    #  "opt_schedule_func": None},
    #
    # {"max_rounds": 10000, "n_round_samples": 8192, "batch_size": 1024, "init_lr": 0.001,
    #  "opt_schedule_func": None},
    #
    # {"max_rounds": 10000, "n_round_samples": 8192, "batch_size": 2048, "init_lr": 0.001,
    #  "opt_schedule_func": None},
)

class check_model(unittest.TestCase):
    RESULT_DIR = 'result-autotest'
    N_VALIDATION = 10000
    TEST_BASE_DIR = 'tmp-autotest'

    def setParameters(self, max_rounds, n_round_samples, batch_size, init_lr, opt_schedule_func):
        self.n_max_rounds = max_rounds
        self.n_round_samples = n_round_samples
        self.batch_size = batch_size
        self.lr = init_lr
        self.opt_schedule_func = opt_schedule_func




    def setUp(self):

        #设置开始时间
        gl.set_value("start_time", int(time.time()))



        if gl.get_value("use_cuda"):
            #self.TEST_BASE_DIR = 'C:/tmp/'
            print("Hello User: using cuda~")
            self.use_cuda = True
        else:
            self.use_cuda = False

        self.seed = 0
        #self.use_cuda = True
        #self.batch_size = 256
        self.test_batch_size = 8192
        #self.lr = 0.001 #学习率,上传的程序没有修改成功
        #self.n_max_rounds = 0
        #self.n_round_samples = 1600 #随机抽取的样本数 #df:1600

        self.testbase = self.TEST_BASE_DIR
        self.testworkdir = os.path.join(self.testbase, 'competetion-test', str(gl.get_value('start_time')))
        self.testIntRound = 100 #测试间隔
        self.savemodel_interval = 50 #保存模型间隔

        self.now_round = 0
        self.outputdir = os.path.join(self.RESULT_DIR, str(gl.get_value('start_time'))) #result/time/
        self.outputdir_model = os.path.join(self.outputdir, 'model')
        self.outputdir_result = os.path.join(self.outputdir, 'result')

        gl.set_value("round_savemodel_int", self.savemodel_interval)

        if not os.path.exists(self.RESULT_DIR):
            os.makedirs(self.RESULT_DIR)

        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)

        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

        if not os.path.exists(self.outputdir_model):
            os.makedirs(self.outputdir_model)
        if not os.path.exists(self.outputdir_result):
            os.makedirs(self.outputdir_result)

        # 初始化参数保存
        # self.n_max_rounds = max_rounds
        # self.n_round_samples = n_round_samples
        # self.batch_size = batch_size
        # self.lr = init_lr
        # self.opt_schedule_func = opt_schedule_func

        self.save_txt = "max-round:%d\nn_round_samples:%d\nbatch_size:%d\nlr:%f\nis_opt:%d" % (self.n_max_rounds, self.n_round_samples, self.batch_size, self.lr, int(self.opt_schedule_func is not None))
        print(self.save_txt)
        with open(os.path.join(self.outputdir, "param.txt"), 'w') as fout:
            fout.write(self.save_txt)
            fout.close()
        # if self.opt_schedule_func is not None:
        #     with open(os.path.join(self.outputdir, "param-opt.pkl"), 'w')  as fout:
        #         pickle.dump(self.opt_schedule_func, fout)
        #         fout.close()

        # 输出保存到文本 无效
        # path = os.path.abspath(os.path.dirname(__file__))
        # type = sys.getfilesystemencoding()
        # sys.stdout = Logger(os.path.join(self.RESULT_DIR, "shell_log_" + str(gl.get_value('start_time')) + ".txt"))

        print("@"*30)
        print("@"*30)
        print("Current ID:%d" % (gl.get_value('start_time')) )

        self.init_model_path = os.path.join(self.testworkdir, 'init_model.pkl')
        torch.manual_seed(self.seed)

        # 如果没有初始化模型,那么执行:模型初始化
        if not os.path.exists(self.init_model_path):
            torch.save(FLModel().state_dict(), self.init_model_path) # model.state_dict()其实返回的是一个OrderDict，存储了网络结构的名字和对应的参数

        if gl.get_value("is_local"):
            self.ps = ParameterServer(init_model_path=self.init_model_path, testworkdir=self.testworkdir,
                                      model_obj=FLModel(), learn_rate=self.lr,
                                      output_model_dir=self.outputdir_model,
                                      opt_schedule=self.opt_schedule_func
                                      )
        else:
            self.ps = ParameterServer(init_model_path=self.init_model_path, testworkdir=self.testworkdir, learn_rate=self.lr,
                                      output_model_dir=self.outputdir_model,
                                      opt_schedule=self.opt_schedule_func
                                      )

        self.urd = user_datasets # 为了多参数自动测试 修改
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
                    print(self.save_txt)
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
        with open(os.path.join(self.outputdir_result, saved_filename), 'w') as fout:
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



def suite():
    suite = unittest.TestSuite()
    suite.addTest(check_model('test_check'))

    return suite


# def suite():
#     suite = unittest.TestSuite()
#     if gl.get_value("test_model_path") is not None:
#         suite.addTest(LocalTestModelTestSuit('ModelLocalTest'))
#     else:
#         suite.addTest(FedAveragingGradsTestSuit('test_federated_averaging'))
#     return suite

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
        gl.gl_init()
        gl.set_value("use_cuda", False)
        gl.set_value("is_local", True)
        gl.set_value("test_model_path", None)
        gl.set_value("init_time", int(time.time()))
        gl.set_value("datasetPath", "../../")

        #加载数据集
        user_datasets = UserRoundData() #加载数据

        if len(user_datasets._user_datasets) == 0:
            print("user dataset is empty")
            exit(0)


    except Exception as e:
        print(e)



    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()

    if os.path.exists("./result-autotest/") == False:
        os.makedirs("./result-autotest/")
    sys.stdout = Logger("./result-autotest/%d.txt" % gl.get_value("init_time"))

    unittest.main()
