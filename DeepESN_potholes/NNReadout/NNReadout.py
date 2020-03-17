import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class LinearReadout(nn.Module):
    def __init__(self, in_features, out_features=None, use_dropout=False):
        super(LinearReadout, self).__init__()
        if out_features is None:
            out_features = in_features
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(in_features, out_features)
        self.tanh = nn.Tanh()

    def forward(self, x):
        if self.use_dropout:
            x = self.dropout(x)
        return self.tanh(self.linear(x))


class DilatedConvNet(nn.Module):
    def __init__(self):
        super(DilatedConvNet, self).__init__()
        self.main = nn.Sequential()
        dilations = [2, 4, 8]
        for i, d in enumerate(dilations):
            padd = 56 if i == 0 else 0
            self.main.add_module(str(i), nn.Conv1d(in_channels=200 if i == 0 else 1, out_channels=1, kernel_size=9, dilation=d, padding=padd))
            self.main.add_module(str(i) + '_elu', nn.ELU())
        self.tanh = nn.Tanh()

    def forward(self, input):
        return self.tanh(self.main(input))


class ReadoutModel():
    def __init__(self, in_features, out_features=None, lr=0.001, momentum=0.9, use_dropout=False, plot=False):
        self.in_features = in_features
        # self.model = LinearReadout(in_features, out_features=out_features, use_dropout=use_dropout)
        self.model = DilatedConvNet()
        self.criterion = nn.MSELoss()
        # self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X, y, X_val=[], y_val=[], epochs=50, batch_size=32):
        dataloader_val = zip(X_val, y_val)
        loss_values = []
        self.model = self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            dataloader = zip(X, y)
            for i, data in enumerate(dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = torch.Tensor(data[0]), torch.Tensor(data[1])
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()

            if (epoch % 5 == 4):
                print('|LROUT| [epoch=%d] loss: %.3f' %
                      (epoch + 1, running_loss / len(X)))
            loss_values.append(running_loss / len(X))
        print('|LROUT| Finished Training')

    def predict(self, X):
        with torch.no_grad():
            X = torch.Tensor(X)
            self.model = self.model.eval()
            out = self.model(X)
        return out.numpy()
