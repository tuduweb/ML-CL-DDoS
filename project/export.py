import torch
from torchsummary import summary
import globalvar as gl

class export(object):
    def __init__(self):
        pass

from learning_model import FLModel


def initTorchModel():
    torch_model = FLModel()

    model_pkl_path = gl.get_value("pkl_path")
    if model_pkl_path:
        pass
    else:
        model_pkl_path = 'result/tmp/competetion-test/1606581609/init_model.pkl'

    torch_model.load_state_dict(torch.load(f=model_pkl_path, map_location=torch.device('cpu')))
    torch_model.eval()

    return torch_model


def export_TorchScript(model, x):
    # model = initTorchModel()
    # x = torch.randn(1, 63)
    mod = torch.jit.trace(model, x)
    mod.save("./model-%s.pt" % gl.get_value("start_time"))


def export_Onnx(model, x):
    # model = initTorchModel()
    # x = torch.randn(1, 63)

    with torch.no_grad():
        torch.onnx.export(
            model,
            x,
            "net.onnx",  # 导出onnx名
            opset_version=11,  # 算子集版本
            input_names=['input'],
            output_names=['output'])

import model
def main():
    global_args = gl.get_value("program_args")

    if global_args.onnx == 0:
        print("onnx == 0")
        exit(0)
    elif global_args.onnx == 1:
        export_Onnx(initTorchModel(), torch.rand(1, 63))
    elif global_args.onnx == 2:
        export_TorchScript(initTorchModel(), torch.rand(1, 63))

    pass