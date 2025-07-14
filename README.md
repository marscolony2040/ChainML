# Machine Learning with Options Data

## Description
This repository contains a combination of Python and C code in order to fill in options implied volatility and greeks for options which do not work with the Trinomial Tree model; PyTorch is used as the Machine Learning library to compute these predictions. The training and testing data are based on whether the Implied Volatility (IV) are able to generate a numerical value or a nan. If the IV is nan, they are pushed to the testing set to predict the iv and greeks. The file xotic.py only predicts the IV while the file xotic_.py predicts all of the Greeks as well. The red are the standard values and the green are the predicted values.

## Video Link
[YouTube Video](https://youtu.be/5HAkJGwVitg?feature=shared)

## Preview
![alt](https://github.com/MoQuant/ChainML/blob/main/P.png)
