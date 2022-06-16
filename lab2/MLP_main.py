# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021

@author: AM4
"""
import pandas as pd
import numpy as np
from neural import MLP


df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
for i in range(len(y)):
    if y[i] == "Iris-setosa":
        y[i] = 0.1
    if y[i] == "Iris-versicolor":
        y[i] = 0.4
    if y[i] == "Iris-virginica":
        y[i] = 0.8
y = y.astype(float).reshape(-1,1)

X = df.iloc[0:100, [0, 1, 2, 3]].values

inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 10 # задаем число нейронов скрытого (А) слоя 
outputSize = 3 # количество выходных сигналов равно количеству классов задачи

iterations = 10000
learning_rate = 0.1

net = MLP(inputSize, outputSize, learning_rate, hiddenSizes)

# обучаем сеть (фактически сеть это вектор весов weights)
i = 0
while sum(abs(np.where(y-(net.predict(X) > 0.5), 1, 0))) != 0:
    i += 1
    net.train(X, y)

# считаем ошибку на обучающей выборке
pr = net.predict(X)
print(sum(abs(y-(pr>0.5))))