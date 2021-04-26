import random
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim as optim
from load_data import *
from utils import *
from stgcn import *


torch.backends.cudnn.deterministic = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

matrix_path = "../W.csv"
data_path = "../traffic.txt"
save_path = "save/model.pt"

day_slot = 288
n_train, n_val, n_test = 34, 5, 5


n_his = 12
n_pred = 3
n_route = 228
Ks, Kt = 3, 3
blocks = [[1, 32, 64], [64, 32, 128]]
drop_prob = 0

batch_size = 50
epochs = 50
lr = 1e-3

W = load_matrix(matrix_path)
L = scaled_laplacian(W)
Lk = cheb_poly(L, Ks)
Lk = torch.Tensor(Lk.astype(np.float32)).to(device)

train, val, test = load_data(data_path, n_train * day_slot, n_val * day_slot)
scaler = StandardScaler()
# train = scaler.fit_transform(train)
# val = scaler.transform(val)
# test = scaler.transform(test)

x_train, y_train = data_transform(train, n_his, n_pred, day_slot, device)
x_val, y_val = data_transform(val, n_his, n_pred, day_slot, device)
x_test, y_test = data_transform(test, n_his, n_pred, day_slot, device)

train_data = TensorDataset(x_train, y_train)
train_iter = DataLoader(train_data, batch_size, shuffle=True)
val_data = TensorDataset(x_val, y_val)
val_iter = DataLoader(val_data, batch_size)
test_data = TensorDataset(x_test, y_test)
test_iter = DataLoader(test_data, batch_size)

loss = nn.MSELoss()
model = STGCN(Ks, Kt, blocks, n_his, n_route, Lk, drop_prob).to(device)
optimizer = optim.RMSprop(model.parameters(), lr=lr)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

min_val_loss = np.inf
for epoch in range(1, epochs + 1):
    l_sum, n = 0.0, 0
    model.train()
    for x, y in train_iter:
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    scheduler.step()
    val_loss = evaluate_model(model, loss, val_iter)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
    print("epoch", epoch, ", train rmse:", torch.sqrt(l_sum / n), ", validation rmse:", torch.sqrt(val_loss))
