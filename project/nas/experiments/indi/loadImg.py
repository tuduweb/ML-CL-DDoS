import cv2
import os
import sys

from torchvision import transforms
from PIL import Image
from indi import EvoCNNModel
from torchsummary import summary

shape = (28, 28)

preprocess_transform = transforms.Compose([
    # transforms.Resize(28, 28),
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor(),
    # https://zhuanlan.zhihu.com/p/414242338
    transforms.Normalize(
        mean=[0.5],
        std=[0.5],
    )
    ])

import torch

if __name__ == '__main__':

    # 环境初始化
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #im = cv2.imread(sys.argv[1], 0)
    #im = cv2.resize(im, shape)
    imgPath = "E:/mnist/9990-label-7.png"
    if len(sys.argv) > 1:
        imgPath = sys.argv[1]

    _img = Image.open(imgPath)

    print(_img.mode,_img.size,_img.format)
    for i in range(0, _img.size[0]):
        outputLine = []
        for j in range(0, _img.size[1]):
            pix = _img.getpixel((j, i))
            outputLine.append(str(pix).zfill(3) if pix else "   ")
        print(outputLine)

    img = preprocess_transform(_img)

    img_flatten = img.flatten()
    print(img_flatten)

    model = EvoCNNModel()
    model.load_state_dict(torch.load('./indi00031_00028.pt',  map_location=device), False)

    batch_size = 1  # 批处理大小
    input_shape = (1, 32, 32)  # 输入数据,我这里是灰度训练所以1代表是单通道，RGB训练是3，128是图像输入网络的尺寸
    x = torch.randn(batch_size, *input_shape).cpu()  # 生成张量

    model.eval().cpu()

    outputs = model(x) #img
    # unsqueeze(0), 升维
    imgOutputs = model(img.unsqueeze(0)) #img
    print("outputs\n", imgOutputs)
    _, imgLabel_ = torch.max(imgOutputs, 1)
    print(imgLabel_)

    summary(model.cpu(), (1, 32, 32), batch_size=16)

    print("end")