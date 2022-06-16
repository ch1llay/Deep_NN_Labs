
import pandas as pd
import numpy as np
from neural import MLP


df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values

Y = np.zeros((y.shape[0], 3), dtype=np.float64)
for i in range(len(y)):
    if y[i] == "Iris-setosa":
        Y[i][0] = 1
    if y[i] == "Iris-versicolor":
        Y[i][1] = 1
    if y[i] == "Iris-virginica":
        Y[i][2] = 1
Y = Y.astype(np.float64).reshape(-1,3)

X = df.iloc[0:100, [0, 1, 2, 3]].values.astype(np.float64)

inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 10 # задаем число нейронов скрытого (А) слоя 
outputSize = 3 # количество выходных сигналов равно количеству классов задачи

iterations = 10000
learning_rate = 0.1

net = MLP(inputSize, outputSize, learning_rate, hiddenSizes)

# обучаем сеть (фактически сеть это вектор весов weights)
i = 0
for i in range(iterations):
    if i % 10000 == 0:
        print(i)
    net.train(X, Y)

# считаем ошибку на обучающей выборке
pr = net.predict(X)
print(sum(abs(Y-(pr>0.5))))