
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("Is a gpu available: " + str(torch.cuda.is_available()))
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.get_device_name(device=device))



#dataset is 94312
#X = torch.load(args.title + "/x_tensor.pd").to(device)
#Y = torch.load(args.title + "/y_tensor.pd").to(device)
X = torch.load("data/x_tensor.pd")
Y = torch.load("data/y_tensor.pd")

ratio = 0.95

print(X.shape)
print(Y.shape)

X_train = X[:round(len(X)*ratio)]
Y_train = Y[:round(len(Y)*ratio)]
X_test = X[round(len(X)*ratio):]
Y_test = Y[round(len(Y)*ratio):]

f = open("model_info.txt", "a")

model = nn.Sequential(
    nn.Linear(X_train.size()[1], 20),
    nn.ReLU(),
    nn.Linear(20, 20),
    nn.ReLU(),
    nn.Linear(20, 252),
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
        print(f'val_loss {val_loss}')

        if stale_epochs > 20:
            break

        if val_loss.item() - min_loss < 0:
            min_loss = val_loss.item()
            stale_epochs = 0
            torch.save(model.state_dict(), "pytorch_model_best.pth")
        else:
            print("bad")
            stale_epochs += 1
    end = time.time()
    print("Epoch time: " + str(end - start))
      
    print(f'Finished epoch {epoch}, latest loss {loss}')
    losses.append(float(np.mean(batch_loss)))

print("Trained for " +str(epoch) + " epochs",file=f)
print('',file=f)


print("Plotting Loss vs Epochs...")

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')

fig, ax = plt.subplots()
plt.plot(losses,color='orange')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.savefig("plots/loss.png")


f.close()
