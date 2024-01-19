from torch import nn
import  torch
import torch.nn.functional as F

from RNN.fileandtoken import load_data_time_machine
from RNN.手撕循环神经网络 import predict_ch8, train_ch8
from utils import try_gpu

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

num_hiddens = 256
rnn_layer = nn.RNN(len(vocab),num_hiddens)

state = torch.zeros((1,batch_size,num_hiddens))
state.shape

X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, state_new.shape

class RNNModel(nn.Module):

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super().__init__(**kwargs)

        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size

        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)

        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers,batch_size,self.num_hiddens),device=device)
        else:
            return (torch.zeros((self.num_directions * self.rnn.num_layers,batch_size,self.num_hiddens),device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers,batch_size,self.num_hiddens),device=device))

device = try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
# print(predict_ch8('time traveller', 10, net, vocab, device))

num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, device)