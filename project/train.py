import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from preprocess import CompDataset


def user_round_train(X, Y, model, device,batch_size = 320 , debug=False):
    data = CompDataset(X=X, Y=Y)
    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,#320
        shuffle=True,
    )

    model.train()

    correct = 0
    prediction = []
    real = []
    total_loss = 0
    model = model.to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # import ipdb
        # ipdb.set_trace()
        # print(data.shape, target.shape)
        output = model(data)
        loss = F.nll_loss(output, target)
        total_loss += loss
        loss.backward()
        pred = output.argmax(
            dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        prediction.extend(pred.reshape(-1).tolist())
        real.extend(target.reshape(-1).tolist())

    grads = {'n_samples': data.shape[0], 'named_grads': {}} #数据数量
    for name, param in model.named_parameters(): #给出网络层的名字和参数的迭代器
        #print(name)
        #print(param.grad)
        #can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
        grads['named_grads'][name] = param.grad.detach().cpu().numpy()  #cpu类型 #detach会产生一个gradient的copy但是做过剪枝处理，并没有连接任何算子。纯正用来拿数据的。
        #print(param.shape)

    if debug:
        print('Training Loss: {:<10.2f}, accuracy: {:<8.2f}'.format(
            total_loss, 100. * correct / len(train_loader.dataset)))

    return grads
