import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import matplotlib.pyplot as plt

# x**2
x_dat = torch.tensor([-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0], requires_grad=True)
x_dat = x_dat[:,None]
print(x_dat.shape)
y_dat = torch.tensor([25.0,16.0,9.0,4.0,1.0,0.0,1.0,4.0,9.0,16.0,25.0]).reshape(-1,1)
grad  = torch.tensor([-10.0,-8.0,-6.0,-4.0,-2.0,0.0,2.0,4.0,6.0,8.0,10.0]).reshape(-1,1)
hess = torch.tensor([2.0 for i in range(11)]).reshape(-1,1)
print(hess)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
def train(x, y, grad, model, optimizer, der=0):
    model.train()

    # Compute prediction error
    ypred = model(x)
    gradspred, = autograd.grad(ypred, x, 
                           grad_outputs=ypred.data.new(ypred.shape).fill_(1),
                           create_graph=True)
    hesspred, = autograd.grad(gradspred, x, grad_outputs=ypred.data.new(ypred.shape).fill_(1), create_graph=True)
    if der == 0:
        loss = torch.mean((y - ypred) ** 2)
    elif der == 1:
        loss = torch.mean((y - ypred) ** 2 + (gradspred - grad))
    elif der == 2:
        loss = torch.mean((y - ypred) ** 2 + 1.0*(gradspred - grad) ** 2 + 1.0*(hess - hesspred)**2)
    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item(), gradspred, hesspred
    #loss = loss.item()
    #print(f"loss: {loss:>7f}")

epochs = 5000
for t in range(epochs):
    err, gradspred, hesspred = train(x_dat, y_dat, grad, model, optimizer, der = 2)
    if (t+1)%5000  == 0:
        print(f"Epoch {t+1}\n-------------------------------")
        print(err)
        print(gradspred)
        print(hesspred)
print("Done!")

xs = np.linspace(-10,10,500, dtype=float)
xs_t = torch.tensor(xs, dtype=float, requires_grad=False)
xs_t = xs_t[:,None]
y_pred = model(xs_t.float())

plt.figure()
plt.plot(x_dat.detach().numpy(), y_dat.numpy(), "ko")
plt.plot(xs, y_pred.detach().numpy(), "b-")
plt.show()
