
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import hist
from hist import Hist

print("Is a gpu available: " + str(torch.cuda.is_available()))
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.get_device_name(device=device))



#dataset is 94312
#X = torch.load(args.title + "/x_tensor.pd").to(device)
#Y = torch.load(args.title + "/y_tensor.pd").to(device)
X = torch.load("x_tensor.pd")
Y = torch.load("y_tensor.pd")

ratio = 0.95

print(X.shape)
print(Y.shape)

X_train = X[:round(len(X)*ratio)]
Y_train = Y[:round(len(Y)*ratio)]
X_test = X[round(len(X)*ratio):]
Y_test = Y[round(len(Y)*ratio):]

f = open("model_info.txt", "a")

model = nn.Sequential(
    nn.Linear(X_train.size()[1], 40),
    nn.ReLU(),
    nn.Linear(40, 40),
    nn.ReLU(),
    nn.Linear(40, 40),
    nn.ReLU(),
    nn.Linear(40, 20),
    nn.Sigmoid()
)

print("",file=f)
print(model, file=f)
print("",file=f)


print("Loaded tensors, training data size " + str(len(X_train)) + ", test data size " + str(len(X_test)) )
print("Training data size " + str(len(X_train)) + ", test data size " + str(len(X_test)), file=f )
print("",file=f)

loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 1000
batch_size = 2000

min_loss, stale_epochs = 100.00, 0

losses = []
test_losses = []

import time

for epoch in range(n_epochs):
    start = time.time()
    batch_loss = []
    for i in range(0, len(X_train), batch_size):
        Xbatch = X_train[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = Y_train[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch[:,:,0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        batch_loss.append(loss.item())
        output = model(X_test)

        val_loss = loss_fn(output, Y_test[:,:,0])
        # print(f'val_loss {val_loss}')

        if stale_epochs > 20:
            break

        if val_loss.item() - min_loss < 0:
            min_loss = val_loss.item()
            stale_epochs = 0
            torch.save(model.state_dict(), "pytorch_model_best.pth")
        else:
            stale_epochs += 1
    end = time.time()
    print("Epoch time: " + str(end - start))
      
    print(f'Finished epoch {epoch}, latest loss {loss}')
    losses.append(float(np.mean(batch_loss)))
    y_pred_test = model(X_test)
    loss = loss_fn(y_pred_test, Y_test[:,:,0])
    test_losses.append(loss.item())


print("Trained for " +str(epoch) + " epochs",file=f)
print('',file=f)


print("Plotting Loss vs Epochs...")

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')

fig, ax = plt.subplots()
plt.plot(losses,color='orange',label='Train Loss')
plt.plot(test_losses,color='blue',label='Test Loss')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.savefig("plots/loss.png")


model.load_state_dict(torch.load("pytorch_model_best.pth"))

f = open("model_info.txt", "a")


with torch.no_grad():
    y_pred = model(X_test)

nn_rounded = y_pred.argmax(1)

m = torch.zeros(y_pred.shape).scatter(1, nn_rounded.unsqueeze(1), 1.0)

truth_rounded = Y_test[:,:,0].argsort(descending=True)[:,:2]


# print(y_pred.round()[0])
# print(Y_test[0])
# print(y_pred.round()[0] == Y_test[0])
# print((y_pred.round()[0] == Y_test[0]).float().mean())
# print()
# print(y_pred.round()[1])
# print(Y_test[1])
# print(y_pred.round()[1] == Y_test[1])
# print((y_pred.round()[1] == Y_test[1]).float().mean())
# print()
# print(y_pred.round()[2])
# print(Y_test[2])
# print(y_pred.round()[2] == Y_test[2])
# print((y_pred.round()[2] == Y_test[2]).float().mean())
# print()
# print(y_pred.round()[3])
# print(Y_test[3])
# print(y_pred.round()[3] == Y_test[3])
# print((y_pred.round()[3] == Y_test[3]).float().mean())
# print()

g1 = nn_rounded == truth_rounded[:,0]
g2 = nn_rounded == truth_rounded[:,1]
print()
print()

model_accuracy = (torch.any(torch.stack([g1,g2],-1),-1)).float().mean()
print(f"Accuracy {model_accuracy}")
#print(f"Model Accuracy {model_accuracy}",file=f)

#print(f"Mass Asymmetry Minimization Accuracy {masym_accuracy}",file=f)
print('',file=f)



print("Plotting Triplet Mass...")
M = torch.load("m_tensor.pd")
M_train = M[:round(len(M)*ratio)]
M_test = M[round(len(M)*ratio):]

plt.clf()

ax = hist.axis.Regular(50, 0, 1000, flow=False, name="x")
cax = hist.axis.StrCategory(["Neural Network", "Truth"], name="c")

full_hist = Hist(ax,cax)


for i,masses in enumerate(M_test):

    full_hist.fill(x=masses[truth_rounded[i][0]],c="Truth")
    full_hist.fill(x=masses[truth_rounded[i][1]],c="Truth")
    full_hist.fill(x=masses[nn_rounded[i]],c="Neural Network")

s = full_hist.stack("c")
s.plot()
plt.legend()
plt.xlabel("Invariant Mass of the Heavier of the Triplets [GeV]")
plt.ylabel("")

plt.savefig("plots/inv_mass.png")

f.close()
