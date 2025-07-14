'''
    This code predicts options greeks using PyTorch
'''

import numpy as np
import json
import time
import datetime
import matplotlib.pyplot as plt
from matplotlib import rcParams
import ctypes
import torch
import torch.nn as nn
import torch.optim as optim

rcParams['figure.autolayout'] = True

# Define Neural Network
class NeuralNetwork(nn.Module):

    def __init__(self, inputs, outputs):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(inputs, 50)
        self.layer3 = nn.Linear(50, outputs)
        self.relu = nn.ReLU()

    # Forward propigation
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

# Load all C functions to calculate the Greeks
cd = ctypes.c_double
ci = ctypes.c_int 

exotic = ctypes.CDLL("./comp.so")
exotic.IV.argtypes = (cd, cd, cd, cd, cd, cd, ci, ci)
exotic.IV.restype = cd

exotic.Vanna.argtypes = (cd, cd, cd, cd, cd, cd, ci, ci)
exotic.Vanna.restype = cd

exotic.DeltaDecay.argtypes = (cd, cd, cd, cd, cd, cd, ci, ci)
exotic.DeltaDecay.restype = cd

exotic.Volga.argtypes = (cd, cd, cd, cd, cd, cd, ci, ci)
exotic.Volga.restype = cd

exotic.Veta.argtypes = (cd, cd, cd, cd, cd, cd, ci, ci)
exotic.Veta.restype = cd

exotic.GammaDecay.argtypes = (cd, cd, cd, cd, cd, cd, ci, ci)
exotic.GammaDecay.restype = cd

exotic.Zomma.argtypes = (cd, cd, cd, cd, cd, cd, ci, ci)
exotic.Zomma.restype = cd

exotic.Speed.argtypes = (cd, cd, cd, cd, cd, cd, ci, ci)
exotic.Speed.restype = cd

exotic.Ultima.argtypes = (cd, cd, cd, cd, cd, cd, ci, ci)
exotic.Ultima.restype = cd

# Load the options chain and inputs for the trinomial tree
class Data:

    def __init__(self, ticker='SPY'):
        # Setting parameters and importing options chain json file
        self.stock_price = 554.42
        self.div_yield = 0.0129
        self.risk_free = 0.0521
        self.dataset = json.loads(open('opChain.json','r').read())

    # Converts datetime to timestamp and returns the inputs to the Trinomial Tree
    def cleanup(self, bounds=60):
        the_date = '2024-09-20'
        calls = self.dataset['options'][the_date]['c']
        puts = self.dataset['options'][the_date]['p']
        call_options = []
        put_options = []
        for i, j in calls.items():
            if float(i) >= self.stock_price - bounds and float(i) <= self.stock_price + bounds:
                call_options.append([float(i), float(j['l'])])
        for i, j in puts.items():
            if float(i) >= self.stock_price - bounds and float(i) <= self.stock_price + bounds:
                put_options.append([float(i), float(j['l'])])
        the_time = time.mktime(datetime.datetime.strptime(the_date, '%Y-%m-%d').timetuple())
        delta = (int(the_time) - int(time.time()))/(60*60*24*365)
        
        # S, K, rf, div, t, op
        return self.stock_price, self.risk_free, self.div_yield, delta, call_options, put_options

# Load the data class and fill parameters
data = Data()
S, r, q, T, calls, puts = data.cleanup()
steps = 60
optype = 0

trainIV = []
trainIVY = []
testIV = []

# Build training set if implied volatility is not equal to zero, else add to testing set to be predicted
for strike, price in calls:
    iv = exotic.IV(price, S, strike, r, q, T, steps, optype)
    if iv != 0:
        trainIV.append([price, S, strike, r, q, T])
        temp = [iv]
        
        # Loop through greek functions from C
        for j, metric in enumerate((exotic.Vanna, exotic.DeltaDecay, exotic.Volga, exotic.Veta, exotic.GammaDecay, exotic.Zomma, exotic.Speed, exotic.Ultima)):
            z = metric(S, strike, r, q, iv, T, steps, optype)
            temp.append(z)
            
        trainIVY.append(temp)
    else:
        testIV.append([price, S, strike, r, q, T])

# Define ML Parameters and Neural Network
learning_rate = 0.001
epochs = 2000

# Instead of 1 output there are 9 outputs which include all of the greeks
model = NeuralNetwork(6, 9)

# Set Adam optimizer and loss criterion
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Convert training and testing sets to a PyTorch tensor
INX = [torch.tensor(i, dtype=torch.float32) for i in trainIV]
OUTY = [torch.tensor(i, dtype=torch.float32) for i in trainIVY]
TESTX = [torch.tensor(i, dtype=torch.float32) for i in testIV]

INX = torch.stack(INX)
OUTY = torch.stack(OUTY)
TESTX = torch.stack(TESTX)

# Train the model
print("Training the model")
for epoch in range(epochs):
    # Prediction
    output = model(INX)

    # Loss function
    loss = criterion(output, OUTY)

    # Back Propigation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print number of epochs left
    print(f"Epochs Left: {epochs - epoch} | Loss: {loss.item()}")

# Generate options implied volatilty and greek predictions
print("Testing the model")
with torch.no_grad():
    testout = model(TESTX)

# Set up the plot for all of the IV and Greeks
print("Plotting the model")
names = ('IV','Vanna','DeltaDecay','Volga','Veta','GammaDecay','Zomma','Speed','Ultima')

# Take predictions and set them back to numpy format from a PyTorch tensor
test_ivY = testout.numpy()

fig = plt.figure(figsize=(11, 7))
ax = [fig.add_subplot(u) for u in range(331, 340)]

# Extract training strikes and testing strikes which are in the 3rd column hence index 2
trainK = np.array(trainIV)[:, 2]
testK = np.array(testIV)[:, 2]

# Standard greeks which were not predicted
train_ivY = np.array(trainIVY)

# Plot the standard greeks in red and the predicted greeks in limegreen
for i, name in enumerate(names):
    ax[i].scatter(trainK, train_ivY[:, i], color='red', s=7)
    ax[i].scatter(testK, test_ivY[:, i], color='green', s=7)
    ax[i].set_title(name)

plt.show()
