import mlddos_pnnx
import numpy as np
import torch

if __name__ == '__main__':

    netdata = mlddos_pnnx.test_inference()

    print(netdata)
    print(torch.sum(netdata))
    print(torch.max(netdata))
    pass