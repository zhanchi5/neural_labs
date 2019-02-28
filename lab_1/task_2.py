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
        value = NOT(v4)  # - - my function
        # v1 = NOT(element[1])
        # v2 = OR(v1, element[3])
        # v3 = AND(v2, element[0])
        # v4 = AND(element[0], element[2])
        # v5 = OR(v3, v4)
        # value = NOT(v5)
        # v1 = element[0]
        # v2 = element[1]
        # v3 = AND(v1, v2)
        # v4 = NOT(v3)
        # v5 = AND(element[2], element[3])
        # value = AND(v4, v5)
        target_function.append(int(value))
    for element in X:
        element.insert(0, 1)  # fictional x0 for delta-rule
    for i in range(len(X)):
        F.append((np.array(X[i]), target_function[i]))
    return F


def find_net(inputs, weights):
    return sum(map(lambda weights, inputs: weights * inputs, weights, inputs))


def find_out(net):
    return 0.5 * (np.tanh(net)+1)


def find_y(out):
    if out < 0.5:
        return 0
    else:
        return 1


def find_delta(core_outputs, y):
    return core_outputs - y


def find_der(net):
    temp = find_out(net)
    return (np.cosh(temp)**2)/2


def find_delta_weights(eta, delta, net, current_input):
    return np.dot(eta * delta * find_der(net), current_input)


def weights_correction(weights, eta, delta, net, current_input):
    return [round(a+b, 3) for a, b in zip(weights, find_delta_weights(eta, delta, net, current_input))]


def error_graph(error_values, k_s):
    trace = go.Scatter(
        x=k_s,
        y=error_values,
        mode='lines+markers',
        name='Суммарная ошибка по эпохам обучения (логическая ФА)'
    )
    offline.plot({'data': [trace]}, image='png', image_filename='task_2')
    return


if __name__ == "__main__":
    F, W = initialize()
    T = np.array([y[1] for y in F])
    X = np.array([x[0] for x in F])

    k_s = []
    epoch = 0

    Error = 1
    errors = []
    eta = 0.3
    while Error != 0:
        Error = 0
        y = np.zeros(len(T))
        for i in range(len(T)):
            net = find_net(X[i], W)
            y[i] = find_y(find_out(net))
            W = weights_correction(W, eta, find_delta(T[i], y[i]), net, X[i])
            Error += abs(y[i] - T[i])
        errors.append(Error)
        print(f'{epoch}; {W}; {y}; {Error}', end="\n")
        k_s.append(epoch)
        epoch += 1
    error_graph(error_values=errors, k_s=k_s)
