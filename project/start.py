import export
import model
import argparse
import unittest
import globalvar as gl
import time
import sys
import os
from utils.Logger import Logger
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='bin Neural Network pytorch Arch')
#parser.add_argument('tool')
parser.add_argument('--cuda', '-c', help='是否应用cuda', default=0)
parser.add_argument('--onnx', '-o', help='onnx', default=0)
parser.add_argument('--local', '-l', help='是否在本地运行', default=0)
parser.add_argument('--test', '-t', help='测试模型', default=0)
parser.add_argument('--model_batch_test', '-bt', help='键入测试模型ID', default=0)
parser.add_argument('--model_batch_save', '-bs', help='GPU模型转换为CPU', default=0)
#parser.add_argument('--body', '-b', help='body 属性，必要参数', required=False)

parser.add_argument('--pkl_path', '-pkl', help='pkl点保存地址', default=0, type=str)
parser.add_argument('--dataset', '-datapath', help='数据集地址', default="", type=str)

parser.add_argument('--sys_test', '-st', help='测试类别:如1为读取最小的dataset...', default=0, type=int)

parser.add_argument('--mode', '-m', help='模式', default="normal", type=str)

global_args = parser.parse_args()

if __name__ == '__main__':
    # 参数相关
    # TODO: 更高级的方式就是根据参数类型来

    # args
    try:
        gl.gl_init()
        gl.set_value("program_args", global_args) # 设置程序参数到全局
        gl.set_value("pwd", sys.path[0])
        gl.set_value("sys_test", global_args.sys_test) # 测试模式: 如, 只加载一个数据集等
        gl.set_value("mode", global_args.mode) # 运行模式
        # config

        gl.set_value("round_savemodel_int", 100)

        gl.set_value("dataset_path",
                     global_args.dataset if global_args.dataset != "" else "../"
                     , "数据集目录")
        gl.set_value("train_dataset_path",
                     os.path.join(gl.get_value("dataset_path"), "./train/")
                     , "训练数据集目录")
        gl.set_value("test_dataset_path",
                     os.path.join(gl.get_value("dataset_path"), "./new_test/1606549610-682356.pkl")
                     , "测试集地址")

        gl.set_value("default_output_path", "", "默认输出目录")
        # 当前目录

        # path = os.path.abspath(os.path.dirname(__file__))
        # type = sys.getfilesystemencoding()

        # 是否使用GPU模式
        gl.set_value("use_cuda", False if global_args.cuda == 0 else True)
        gl.set_value("is_local", False if global_args.local == 0 else True)

        # 推理, 测试模式
        gl.set_value("test_model_path",
                     None if global_args.test == 0 else global_args.test)
        gl.set_value("test_model_path_id",
                     None if global_args.model_batch_test == 0 else int(global_args.model_batch_test))

        # 工具: 模型转换
        gl.set_value("model_convert_to_gpu",
                     None if global_args.model_batch_save == 0 else int(global_args.model_batch_save))

        # 工具: pnnx

        # initPkl: 程序初始化的模型
        gl.set_value("pkl_path", None if global_args.pkl_path == "" else global_args.pkl_path)
    except Exception as e:
        print(e)
        exit(-1)

    # tools
    if gl.get_value("mode") == "tool":
        # 暂时只有导出的一些工具.. 所以直接用export
        export.main()
        exit(0)

    outputGroupName = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    gl.get_value("default_output_path")

    savePath = os.path.join(gl.get_value("pwd"), "./result-autotest/")

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    resultOutputPath = os.path.join(savePath, outputGroupName)
    if not os.path.exists(
        resultOutputPath
    ):
        os.makedirs(resultOutputPath)

    gl.set_value("output_path", resultOutputPath)

    sys.stdout = Logger(
        os.path.realpath(os.path.join(resultOutputPath, "./train_summary.log"))
    ) # gl.get_value("init_time")
    writer = SummaryWriter(os.path.join(resultOutputPath, "logs"))  # 事件存放路径
    gl.set_value("tb_writer", writer) # None

    # config
    gl.set_value("model_config", {
        "batch_size": 2048,
        "test_batch_size": 8192,
        "lr": 0.001,
        "n_max_rounds": 10000,
        "log_interval": 10,
        "n_round_samples": 8192,  # 抽样去
        "testInterRound": 100
    })

    # other
    if gl.get_value("mode") == "normal":

        model.main()

    elif gl.get_value("mode") == "suite":

        unittest.main()

    # TODO: catch exit
    writer.close()
