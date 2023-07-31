import random
import numpy as np
import copy
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from two_moon_util import adapt_target, train_target


EPOCHS = 30
SEED = 2023
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
# =============== source data ===============
moon = make_moons(n_samples=600, shuffle=True, noise=0.08, random_state=2023)
x, y = moon
plt.scatter(x[:,0], x[:,1], c=y, cmap='jet', marker='.')

batch_size = 300
train_y = torch.from_numpy(y)
train_x = torch.from_numpy(x).float()

train_ds = torch.utils.data.TensorDataset(train_x, train_y)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

# =============== source model ===============

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # defining 2 linear layers
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(inplace=True)
                                 )
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim)
                                 )
        self.cls = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        feat = self.extract_feat(x)
        out = self.cls(feat)
        return out

    def extract_feat(self, x):
        feat = self.fc1(x)  # activation on hidden layer
        feat = self.fc2(feat)
        return feat


# =============== train source model ===============
model = SimpleNet(2, 10, 2)
optimizer = optim.Adam(model.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()  # zero accumulated gradients
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(loss.item())

    # print loss stats
    print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))

pred_y = model(torch.from_numpy(x).float())
pred_y = pred_y.argmax(1)
plt.scatter(x[:, 0], x[:, 1], c=pred_y.numpy(), cmap='jet', marker='.')


def plot_decision_boundary(pred_func,X,y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy=np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float())
    Z = Z.detach()
    Z = Z.argmax(1).numpy()
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, colors=['gray', 'orange', 'skyblue'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.binary)

model.eval()
plot_decision_boundary(model, x, y)

ori_state_dict = copy.deepcopy(model.state_dict())

# =============== target dataset ===============

theta = np.pi / 6
rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
x_rot = (rot_matrix @ x.T).T

plt.scatter(x[:, 0], x[:, 1], marker='.')
plt.scatter(x_rot[:, 0], x_rot[:, 1], marker='.')

# =============== adapt to target with NRC ===============
bach_size = 64
tar_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_rot).float(), torch.arange(y.shape[0]),  torch.from_numpy(y))
tar_loader = torch.utils.data.DataLoader(tar_dataset, batch_size=batch_size, shuffle=True)

# original accuracy
model.load_state_dict(ori_state_dict)
model.eval()
tar_x = torch.from_numpy(x_rot).float();
tar_y= torch.from_numpy(y)
pred_y = model(tar_x).argmax(1)
acc = float(torch.sum(pred_y==tar_y))/tar_y.shape[0]
print('acc {}'.format(acc))

model = train_target(model, tar_loader, K=5, max_iter=20, alpha=0.0, beta=5.0)

# =============== adapt to target with LP ===============

model.load_state_dict(ori_state_dict)

model = adapt_target(model, x_rot, y, K=5, max_epoch=10)








