import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def to_full(y, num):
    y_full = np.zeros((1, num))
    y_full[0, y] = 1
    return y_full

def sparse_cross(z, y):
    return -np.log(z) if y == 1 else -np.log(1 - z)

def sig_der(z):
    return z * (1 - z)  # Используем значение z, выход сигмоиды

# Ваши данные
dataset = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

# Инициализация переменных
w = np.random.rand(2, 1)  # Веса
b = np.random.rand(1, 1)  # Смещение
loss_arr = []
ALPHA = 0.1
EPOHS = 1000

for epoch in range(EPOHS):
    for data in dataset:
        x = np.array(data[:2]).reshape(1, -1)  # Входные данные
        y = data[2]  # Целевое значение

        # Прямой проход
        t = x @ w + b
        z = sigmoid(t)
        e = sparse_cross(z, y)
        loss_arr.append(e)

        # Обратное распространение
        de_dz = z - y  # Дериватив потерь по z
        de_dt = de_dz * sig_der(z)  # Дериватив потерь по t, используется z
        de_dw = x.T @ de_dt  # Дериватив потерь по весам
        de_db = de_dt  # Дериватив потерь по смещению

        # Обновление весов и смещения
        w -= ALPHA * de_dw  # Обновление весов
        b -= ALPHA * de_db  # Обновление смещения

# Построение графика потерь
loss_arr = np.array(loss_arr).squeeze()  # Приведение к одномерному массиву
plt.plot(loss_arr)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss over Iterations")
plt.show()
