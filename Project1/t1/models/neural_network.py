import torch
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils.preprocessing import *


class NeuralNetwork(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.ll1 = nn.Linear(num_features, 20)
        self.ll2 = nn.Linear(20, 40)
        self.ll3 = nn.Linear(40, 20)
        self.ll4 = nn.Linear(20, 1)

    def forward(self, input_batches):
        out = self.ll1(input_batches)
        out = self.ll2(out)
        out = self.ll3(out)
        out = self.ll4(out)
        return out


def train_and_predict(path: str):
    tau = 7
    epochs = 500
    USE_GPU = True
    dtype = torch.float32
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    X, y = generate_data(path, tau=7)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    X_train_tensor = torch.Tensor(X_train)
    X_valid_tensor = torch.Tensor(X_valid)
    y_train_tensor = torch.Tensor(y_train)
    y_valid_tensor = torch.Tensor(y_valid)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader_train = DataLoader(train_dataset)
    learning_rate = 1e-2
    model = NeuralNetwork(num_features=tau+2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)
            scores = model(x).squeeze(dim=1)
            loss = F.mse_loss(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if t % 10 == 0:
                # print('Iteration %d, loss = %.4f' % (t, loss.item()))
    with torch.no_grad():
        predictions = model(X_valid_tensor)
        print("Root Mean Squared Error: " + str(torch.sqrt(F.mse_loss(predictions, y_valid_tensor))))
