import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# x
#x_dat = torch.tensor([-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0], requires_grad=True)
#x_dat = x_dat[:,None]
#print(x_dat.shape)
#y_dat = torch.tensor([25.0,16.0,9.0,4.0,1.0,0.0,1.0,4.0,9.0,16.0,25.0]).reshape(-1,1)
#grad  = torch.tensor([-10.0,-8.0,-6.0,-4.0,-2.0,0.0,2.0,4.0,6.0,8.0,10.0]).reshape(-1,1)
#hess = torch.tensor([2.0 for i in range(11)]).reshape(-1,1)
#print(hess)

def f(x1, x2):
    return (3*(x1*(x2**2)) + (x1**3) + x2)/400

def df_x1(x1, x2):
    return (2*(x2**2) + 9*(x1**2))/400

def df_x2(x1, x2):
    return (6*x1*x2+1)/400

def d2f_x12(x1, x2):
    return 18*x1/400

def d2f_x22(x1, x2):
    return 6*x1/400

def d2f_x1x2(x1, x2):
    return 6*x2/400

g = np.linspace(-5,5,51)
print(g)
X, Y = np.meshgrid(g, g)
print(np.shape(X))
Z = f(X, Y)
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot_wireframe(X, Y, Z)
#plt.show()

ndensity = 5
g1 = np.linspace(-5,5,ndensity)
X1, Y1 = np.meshgrid(g1, g1)
Z1 = f(X1, Y1).flatten()
#x_dat = torch.tensor()
x_dat = torch.tensor(np.column_stack((X1.flatten(), Y1.flatten())), dtype=torch.float32, requires_grad=True)
y_dat = torch.tensor(f(X1, Y1).flatten(), dtype=torch.float32, requires_grad=False).reshape(-1,1)
grad = torch.tensor(np.column_stack((df_x1(X1, Y1).flatten(), df_x2(X1, Y1).flatten())), dtype=torch.float32)
mn4 = np.column_stack((d2f_x12(X1, Y1).flatten(), d2f_x1x2(X1, Y1).flatten(), d2f_x1x2(X1, Y1).flatten(), d2f_x22(X1, Y1).flatten()))
#print(mn4.reshape((25,2,2)))
hess = torch.tensor(mn4.reshape((ndensity**2,2,2)), dtype=torch.float32)
print(hess.size())

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 10),
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
optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-3)
def train(x, y, grad, hess, model, optimizer, der=0):
    model.train()

    # Compute prediction error
    def closure():
        optimizer.zero_grad()
        ypred = model(x)
        gradspred, = autograd.grad(ypred, x, 
                               grad_outputs=ypred.data.new(ypred.shape).fill_(1),
                               create_graph=True)
        s = gradspred.size()
        hesspred = torch.zeros(s[0], s[1], s[1])
        #for i in range(s[1]):
        #    print(autograd.grad(gradspred[:,i], x, grad_outputs=torch.ones(25), create_graph=True))
        #    hesspred[:,i,:],  = autograd.grad(gradspred[:,i], x, grad_outputs=gradspred.data.new(gradspred.shape).fill_(1), create_graph=True)
        hesspredi = [autograd.grad(gradspred[:,t], x, grad_outputs=gradspred.data.new(s[0]).fill_(1), create_graph=True)[0] for t in range(s[1])]
        hesspred = torch.stack(hesspredi, dim=2)
        #print(hesspred.size())
        #beans = (hess - hesspred)**2
        #beebus = (gradspred - grad)**2
        #seebus = (y-ypred)**2
        #print(seebus.size())
        #print(beebus.size())
        #print(torch.sum(beans, dim=2).size())
        if der == 0:
            loss = torch.sqrt(torch.mean((y - ypred) ** 2))
        elif der == 1:
            loss = torch.sqrt(torch.mean((y - ypred) ** 2 + (gradspred - grad)**2))
        elif der == 2:
            loss = torch.sqrt(torch.mean((y - ypred) ** 2 + torch.sum(1.0*(gradspred - grad) ** 2,dim=1).reshape(-1,1) + torch.sum(1.0*(hess - hesspred)**2, dim=(1,2)).reshape(-1,1)))
        # Backpropagation
        loss.backward()
        return loss
    
    optimizer.step(closure)
    #optimizer.zero_grad()
    #return loss.item(), gradspred, hesspred
    #loss = loss.item()
    #print(f"loss: {loss:>7f}")

def test(x, y, grad, hess, model):
    ypred = model(x)
    gradspred, = autograd.grad(ypred, x, 
                           grad_outputs=ypred.data.new(ypred.shape).fill_(1),
                           create_graph=True)
    s = gradspred.size()
    hesspred = torch.zeros(s[0], s[1], s[1])
    hesspredi = [autograd.grad(gradspred[:,t], x, grad_outputs=gradspred.data.new(s[0]).fill_(1), create_graph=True)[0] for t in range(s[1])]
    hesspred = torch.stack(hesspredi, dim=2)
    loss = torch.sqrt(torch.mean((y - ypred) ** 2))
    gradloss = torch.sqrt(torch.mean(torch.sum(1.0*(gradspred - grad) ** 2,dim=1).reshape(-1,1)))
    hessloss = torch.sqrt(torch.mean(torch.sum(1.0*(hess - hesspred)**2, dim=(1,2)).reshape(-1,1)))

    g1 = np.linspace(-5,5,51)
    X1, Y1 = np.meshgrid(g1, g1)
    #Z1 = f(X1, Y1).flatten()
    #x_dat = torch.tensor()
    x_dat = torch.tensor(np.column_stack((X1.flatten(), Y1.flatten())), dtype=torch.float32, requires_grad=True)
    y_dat = torch.tensor(f(X1, Y1).flatten(), dtype=torch.float32, requires_grad=False).reshape(-1,1)
    gradt = torch.tensor(np.column_stack((df_x1(X1, Y1).flatten(), df_x2(X1, Y1).flatten())), dtype=torch.float32)
    mn4t = np.column_stack((d2f_x12(X1, Y1).flatten(), d2f_x1x2(X1, Y1).flatten(), d2f_x1x2(X1, Y1).flatten(), d2f_x22(X1, Y1).flatten()))
    #print(mn4.reshape((25,2,2)))
    hesst = torch.tensor(mn4t.reshape((51**2,2,2)), dtype=torch.float32)
    ypredt = model(x_dat)
    gradspred, = autograd.grad(ypredt, x_dat, 
                           grad_outputs=ypredt.data.new(ypredt.shape).fill_(1),
                           create_graph=True)
    s = gradspred.size()
    hesspred = torch.zeros(s[0], s[1], s[1])
    hesspredi = [autograd.grad(gradspred[:,t], x_dat, grad_outputs=gradspred.data.new(s[0]).fill_(1), create_graph=True)[0] for t in range(s[1])]
    hesspred = torch.stack(hesspredi, dim=2)
    test_err = torch.sqrt(torch.mean((y_dat - ypredt) ** 2))
    test_grad = torch.sqrt(torch.mean(torch.sum(1.0*(gradspred - gradt) ** 2,dim=1).reshape(-1,1)))
    test_hess = torch.sqrt(torch.mean(torch.sum(1.0*(hesst - hesspred)**2, dim=(1,2)).reshape(-1,1)))
    
    return test_err, test_grad, test_hess, loss, gradloss, hessloss

epochs = 1000
for t in range(epochs):
    #err, gradspred, hesspred = train(x_dat, y_dat, grad, model, optimizer, der = 2)
    train(x_dat, y_dat, grad, hess, model, optimizer, der=0)
    #if (t+1)%1000  == 0:
    #    print(f"Epoch {t+1}\n-------------------------------")
    #    print(err)
    #    print(gradspred)
    #    print(hesspred)
print("Done!")
test_err, test_grad, test_hess, loss, gradloss, hessloss = test(x_dat, y_dat, grad, hess, model)
print(loss.item(), gradloss.item(), hessloss.item())
print(test_err.item(), test_grad.item(), test_hess.item())
g = np.linspace(-5,5,51)
Xt, Yt = np.meshgrid(g, g)
X = Xt.flatten()
Y = Yt.flatten()
xdat = torch.tensor(np.column_stack((X,Y)), dtype=torch.float32)
Z = model(xdat)
#print(X)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x_dat[:,0].detach().numpy(), x_dat[:,1].detach().numpy(), y_dat.numpy(), "ko")
ax.plot_wireframe(Xt, Yt, Z.detach().numpy().reshape(51,51))
plt.show()
