import torch.nn as nn
import torch.nn.functional as F


class FLModel(nn.Module):
    def __init__(self):
        super().__init__()
        #PyTorch的nn.Linear（）是用于设置网络中的全连接层的，需要注意的是全连接层的输入与输出都是二维张量，一般形状为[batch_size, size]，不同于卷积层要求输入输出是四维张量。
        #张量是一个可用来表示在一些矢量、标量和其他张量之间的线性关系的多线性函数
        self.fc1 = nn.Linear(79, 256)
        self.fc5 = nn.Linear(256, 14)
        #in_features指的是输入的二维张量的大小，即输入的[batch_size, size]中的size。
        #out_features指的是输出的二维张量的大小，即输出的二维张量的形状为[batch_size，output_size]，当然，它也代表了该全连接层的神经元个数。

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        #relu:线性整流函数（Rectified Linear Unit, ReLU），又称修正线性单元，是一种人工神经网络中常用的激活函数（activation function），通常指代以斜坡函数及其变种为代表的非线性函数。
        x = self.fc5(x)
        output = F.log_softmax(x, dim=1)
        #log_softmax:输出层的激励函数.把一组评分值转换成一组概率,总概率和为1,保持相对大小关系

        return output
