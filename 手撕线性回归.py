import torch


def synthetic(w, b, nums):
    if not isinstance(w, torch.Tensor):
        w = torch.tensor(w, dtype=torch.float32)


    x = torch.normal(0, 1, (nums, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 1, y.shape)

    return x, y.reshape((-1, 1))

def net(X,w,b):
    '''定义模型'''
    return torch.matmul(X,w) +b

def square_loss(y_hat,y):
    '''计算损失'''
    return (y-y_hat.reshape(y.shape))**2/2


def update_param(params,lr,bach_size):
    '''更新参数'''
    with torch.no_grad():
        for param in params:
            param -=lr*param.grad/bach_size
            param.grad.zero_()


import random

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def data_iter(feature, y, batch_size):
    indices = list(range(len(feature)))
    random.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        inde = indices[i:min(i + batch_size, len(indices) - 1)]

        yield feature[inde], y[inde]

def shousi():
    batch_size = 100
    epochs = 100
    lr = 1e-2
    w,b = torch.normal(0,1,(2,1),requires_grad=True),torch.zeros(1,requires_grad=True)
    w = torch.tensor([0.,0],requires_grad=True)
    feature,labels = synthetic([1,2],1,1000)

    for epoch in range(epochs):
        for x,y in data_iter(feature,labels,batch_size):
            l = square_loss(net(x,w,b),y)
            l.sum().backward()
            sgd([w,b],lr,batch_size)
        with torch.no_grad():
            l = square_loss(net(feature,w,b),labels)
            l = l.sum().item()/len(feature)
            print(f"epoch={epoch+1},loss={l:0.3}")

    print(w,b)

def jianjie():
    from torch import nn
    from torch.utils import data

    net = nn.Sequential(nn.Linear(2,1))
    loss = nn.MSELoss()
    net[0].weight.data.normal_(0,0.01)
    net[0].bias.data.fill_(0)
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)
    feature, labels = synthetic([1, 2], 1, 1000)
    dataset = data.TensorDataset(*(feature,labels))
    loader = data.DataLoader(dataset,batch_size=100,shuffle=True)

    epochs = 100
    for epoch in range(epochs):
        for x,y in loader:
            l = loss(net(x),y)
            trainer.zero_grad()
            l.backward()
            trainer.step()

        l = loss(net(feature),labels)
        l = l.item()
        print(f"epoch = {epoch},loss={l}")
    print(net[0].weight.data,net[0].bias.data)



from sklearn.ensemble import GradientBoostingClassifier
jianjie()

