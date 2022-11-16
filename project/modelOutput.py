import torch

from learning_model import FLModel
from torchsummary import summary

model = FLModel().to('cpu')

print(model)
summary(model, input_size = [(63, 1)], batch_size=1)