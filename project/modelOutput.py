import torch

from learning_model import FLModel
from torchsummary import summary

model = FLModel().to('cpu')

print(model)
summary(model, input_size = [(1, 63)], batch_size=1)