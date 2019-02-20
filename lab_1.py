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
# import plotly
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
        # v1 = OR(element[0], element[1])
        # v2 = AND(v1, element[2])
        # v3 = AND(element[2], element[3])
        # v4 = OR(v2, v3)
        # value = NOT(v4) -- my function
        # v1 = NOT(element[1])
        # v2 = OR(v1, element[3])
        # v3 = AND(v2, element[0])
        # v4 = AND(element[0], element[2])
        # v5 = OR(v3, v4)
        # value = NOT(v5)
        v1 = element[0]
        v2 = element[1]
        v3 = AND(v1, v2)
        v4 = NOT(v3)
        v5 = AND(element[2], element[3])
        value = AND(v4, v5)
        target_function.append(int(value))
    for element in X:
        element.insert(0, 1)  # fictional x0 for delta-rule
    for i in range(len(X)):
        F.append((np.array(X[i]), target_function[i]))
    return F


def unit_step(x): return 0. if x < 0 else 1.


def nnm_BF(X, W, F):
    # F - outputs
    eta = 0.3  # eductation norma
    k = 0
    iterations_number = 25
    W = np.array(W)
    errors = np.ones(len(F))
    for element in X:
        element.insert(0, 1)  # fictional variable x0 added

    pdb.set_trace()
    training_data = []
    for i in range(0, len(F)):
        training_data.append((np.array(X[i]), F[i]))
    predictions = np.ones(len(F))
    # expectations = np.zeros(len(F))
    while k < iterations_number:
        print(f'Era number: {k}')
        print(f'Weights: {W}')

        for i in range(0, len(training_data)):
            # vector for errors (actual - predictions)
            result = np.dot(W, training_data[i][0])
            predicted = unit_step(result)
            predictions[i] = predicted

            expected = training_data[i][1]
            # expectations[i] = expected
            error = expected - predicted
            # errors[i] = error
            W += [eta * error * x for x in X[i]]
        pdb.set_trace()
        k += 1
    print(F)
    return


def training_perceptron(initializer):
    eta = 0.3  # training norma
    k = 0
    iterations_number = 25
    data, W = initializer

    X = [x[0] for x in data]
    Y = [y[1] for y in data]

    y_pred = np.ones(len(data))
    errors = np.ones(len(data))

    while k < iterations_number:
        outputs = []
        print(f'Era number: {k}')
        print(f'Weights: {W}')
        print(f'Errors: {errors} ')
        # print(f'Errors: {errors}')
        # print(f'SUM E: {abs(np.sum(Y-y_pred))}')
        for i in range(0, len(X)):
            net = np.dot(X[i], W)
            y_net = unit_step(net)
            outputs.append(y_net)
            y_pred[i] = y_net
            error = (Y[i] - y_pred[i])**2
            errors[i] = error
            for j in range(0, len(W)):
                W[j] += eta * (Y[i] - y_pred[i]) * X[i][j]  # Updating weights
        k += 1
    pdb.set_trace()


if __name__ == "__main__":
    my_task = initialize()
    training_perceptron(my_task)
    # nnm_BF(X=my_task[0], W=my_task[1], F=my_task[2])
