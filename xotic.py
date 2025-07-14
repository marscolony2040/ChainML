'''
    This code only predicts implied volatility for options which are not
    solveable with a Trinomial Tree
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

# Neural Network Class with PyTorch
class NeuralNetwork(nn.Module):

    def __init__(self, inputs, outputs):
        super(NeuralNetwork, self).__init__()
        # Declare hidden layers
        self.layer1 = nn.Linear(inputs, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, outputs)
        self.relu = nn.ReLU()

    # Forward propigation
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

# Initialize all C functions which calculate the greeks and set their datatypes for inputs and results
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

# Dataset class which opens and parses the options chain
class Data:

    def __init__(self, ticker='SPY'):
        # Setting initial parameters
        self.stock_price = 554.42
        self.div_yield = 0.0129
        self.risk_free = 0.0521

        # Options chain json dataset
        self.dataset = json.loads(open('opChain.json','r').read())
    
    # converts the time to numerical values and returns the parameters to be passed into the trinomial tree
    def cleanup(self, bounds=60):
        the_date = '2024-09-20'
        calls = self.dataset['options'][the_date]['c']
        puts = self.dataset['options'][the_date]['p']
        call_options = []
        put_options = []
        # Sets bounds for call and put options to not deviate Strikes too far from the stock price
        for i, j in calls.items():
            if float(i) >= self.stock_price - bounds and float(i) <= self.stock_price + bounds:
                call_options.append([float(i), float(j['l'])])
        for i, j in puts.items():
            if float(i) >= self.stock_price - bounds and float(i) <= self.stock_price + bounds:
                put_options.append([float(i), float(j['l'])])

        # Timestamp conversion
        the_time = time.mktime(datetime.datetime.strptime(the_date, '%Y-%m-%d').timetuple())
        delta = (int(the_time) - int(time.time()))/(60*60*24*365)
        
        # results) S, K, rf, div, t, op
        return self.stock_price, self.risk_free, self.div_yield, delta, call_options, put_options

# Fetch data
data = Data()
S, r, q, T, calls, puts = data.cleanup()

# Define steps of tree
steps = 50
optype = 0

trainIV = []
trainIVY = []
testIV = []

# Loop through all call options and extract the implied volatility using the C function
for strike, price in calls:
    iv = exotic.IV(price, S, strike, r, q, T, steps, optype)
    if iv != 0:
        # if options are solveable they are put into the training set
        trainIV.append([price, S, strike, r, q, T])
        trainIVY.append(iv)
    else:
        # if options iv is not solveable the data is put into the testing set where the iv will be predicted
        testIV.append([price, S, strike, r, q, T])

# Set up neural network to predict one variable along with Adam optimizer and loss criterion
learning_rate = 0.00001
epochs = 1000

model = NeuralNetwork(6, 1)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Build PyTorch tensors to store the data and convert the numbers to torch.float32
INX = [torch.tensor(i, dtype=torch.float32) for i in trainIV]
OUTY = torch.tensor(trainIVY, dtype=torch.float32)
TESTX = [torch.tensor(i, dtype=torch.float32) for i in testIV]

INX = torch.stack(INX)
OUTY = torch.stack((OUTY,))
TESTX = torch.stack(TESTX)

# Machine Learning 
for epoch in range(epochs):
    # Predict values
    output = model(INX)
    
    # Pass into loss function
    loss = criterion(output, OUTY.T)

    # Backpropigation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the number of epochs left
    print("Epochs Left: ", epochs - epoch)

# Predict values based on test set of options which could not be computed with the Trinomial Tree
with torch.no_grad():
    testout = model(TESTX)

# Do not pull nan implied volatility
impliedVol = [i if i >= 0 else np.nan for i in testout.numpy()[:,0].tolist()]

# Pull strike prices
trainStrikes = np.array(trainIV)[:,2]
testStrikes = np.array(testIV)[:,2]

print(impliedVol)
plt.pause(3)

fig = plt.figure(figsize=(11, 7))
ax = [fig.add_subplot(num) for num in range(331, 340)]

names = ('Vanna','DeltaDecay','Volga','Veta','GammaDecay','Zomma','Speed','Ultima')

# Plot all greeks with their standard values
for i, iv in enumerate(trainIVY):
    print("TrainIV: ", iv)
    K = trainStrikes[i]
    ax[0].scatter(K, iv, color='red', s=7)
    ax[0].set_title('ImpliedVol')
    
    # Loop through each greek function in C
    for j, metric in enumerate((exotic.Vanna, exotic.DeltaDecay, exotic.Volga, exotic.Veta, exotic.GammaDecay, exotic.Zomma, exotic.Speed, exotic.Ultima)):
        z = metric(S, K, r, q, iv, T, steps, optype)
        ax[j+1].scatter(K, z, color='red', s=7)
        ax[j+1].set_title(names[j])
        
    plt.pause(0.00001)


# Plot all of the predicted implied volatility greeks
for i, iv in enumerate(impliedVol):
    print("TestIV: ", iv)
    K = testStrikes[i]
    ax[0].scatter(K, iv, color='green', s=7)
    #ax[0].set_title('ImpliedVol')

    # Loop through each greek function in C
    for j, metric in enumerate((exotic.Vanna, exotic.DeltaDecay, exotic.Volga, exotic.Veta, exotic.GammaDecay, exotic.Zomma, exotic.Speed, exotic.Ultima)):
        z = metric(S, K, r, q, iv, T, steps, optype)
        ax[j+1].scatter(K, z, color='green', s=7)
        #ax[j+1].set_title(names[j])
    
    plt.pause(0.00001)

plt.show()
