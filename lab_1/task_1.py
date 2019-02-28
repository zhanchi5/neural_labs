''' Lab 1
Исследование однослойных нейронных сетей на примере
моделирования булевых выражений

Цель: Исследовать функционирование простейшей нейронной сети (НС) на базе
нейрона с нелинейной функцией активации и ее обучение по правилу Видроу-Хоффа.

Вариант 15.

b_f = NOT((x1 + x2)x3 + x3x4)
f_a: 1,4
'''

from operator import and_, or_, not_
import numpy as np
import plotly.offline as offline
import plotly.graph_objs as go
import pdb


def AND(x1, x2):
    return and_(x1, x2)


def OR(x1, x2):
    return or_(x1, x2)


def NOT(x):
    return not_(x)


def initialize():
    n = 4  # variables number
    W = np.zeros(n + 1)  # [0., 0., 0., 0., 0.]
    X = []
    i = 0
    while i < 2**int(n):
        x = list(format(i, f'0{n}b'))
        x = [int(n) for n in x]
        X.append(x)
        i += 1
    F = get_F(X)
    return F,  W


def get_F(X):
    F = list()
    target_function = []
    for element in X:
        v1 = OR(element[0], element[1])
        v2 = AND(v1, element[2])
        v3 = AND(element[2], element[3])
        v4 = OR(v2, v3)
        value = NOT(v4)
        target_function.append(int(value))
    for element in X:
        element.insert(0, 1)  # fictional x0 for delta-rule
    for i in range(len(X)):
        F.append((np.array(X[i]), target_function[i]))
    return F


def unit_step(x): return 0. if x < 0 else 1.


def training_perceptron(initializer, eta=0.3):
    k = 0
    data, W = initializer

    k_s = []  # Era numbers for graph
    sum_errors = []  # Sum Error numbers by each Era for graph
    X = [x[0] for x in data]
    Y = [y[1] for y in data]

    y_pred = np.ones(len(data))
    errors = np.ones(len(data))

    while np.sum(errors) != 0:
        outputs = []
        print(f'{k}', end='; ')
        if k < 10:
            print(f' {W}', end='; ')
        else:
            print(f'{W}', end='; ')
        for i in range(0, len(X)):
            net = np.dot(X[i], W)
            y_net = unit_step(net)
            outputs.append(y_net)
            y_pred[i] = y_net
            error = Y[i] - y_pred[i]

            for j in range(0, len(W)):
                W[j] += eta * error * X[i][j]  # Updating weights
        for i in range(0, len(Y)):
            errors[i] = (Y[i] - y_pred[i]) ** 2

        print(f'{outputs}', end='; ')
        sum_errors.append(sum(errors))
        k_s.append(k)
        print(f'{np.sum(errors)}', end='\n')
        k += 1

    error_graph(error_values=sum_errors, k_s=k_s)
    return


def error_graph(error_values, k_s):
    trace = go.Scatter(
        x=k_s,
        y=error_values,
        mode='lines+markers',
        name='Суммарная ошибка по эпохам обучения (пороговая ФА)'
    )
    offline.plot({'data': [trace]}, image='png', image_filename='task_1')
    return


if __name__ == "__main__":
    my_task = initialize()
    training_perceptron(my_task)
