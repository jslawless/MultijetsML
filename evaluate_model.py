
import hist
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from hist import Hist

print(torch.cuda.is_available())

#dataset is 94312
X = torch.load("data/x_tensor.pd")
Y = torch.load("data/y_tensor.pd")

ratio = 0.95

X_train = X[:round(len(X)*ratio)]
Y_train = Y[:round(len(Y)*ratio)]
X_test = X[round(len(X)*ratio):]
Y_test = Y[round(len(Y)*ratio):]

model = nn.Sequential(
    nn.Linear(X_train.size()[1], 20),
    nn.ReLU(),
    nn.Linear(20, 20),
    nn.ReLU(),
    nn.Linear(20, 252),
    nn.Sigmoid()
)

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

f.close()

print("Plotting Triplet Mass...")
M = torch.load("data/m_tensor.pd")
M_train = M[:round(len(M)*ratio)]
M_test = M[round(len(M)*ratio):]


ax = hist.axis.Regular(40, 0, 2250, flow=False, name="x")
cax = hist.axis.StrCategory(["Neural Network", "Mass Asymmetry", "Truth"], name="c")

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
