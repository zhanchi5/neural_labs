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
    while i < 2**(int(n) + 1):
        x = list(format(i, f'0{n+1}b'))
        x = [int(n) for n in x[1:]]
        X.append(x)
        i += 1
    middle = int(len(X) / 2)
    X = X[middle:]
    F = get_F(X)
    return X,  W, F


def get_F(X):
    F = list()
    #  import pprint; pprint.pprint(X)
    for element in X:
        # v1 = OR(element[0], element[1])
        # v2 = AND(v1, element[2])
        # v3 = AND(element[2], element[3])
        # v4 = OR(v2, v3)
        # value = NOT(v4) -- my function
        v1 = NOT(element[1])
        v2 = OR(v1, element[3])
        v3 = AND(v2, element[0])
        v4 = AND(element[0], element[2])
        v5 = OR(v3, v4)
        value = NOT(v5)
        #v1 = element[0]
        #v2 = element[1]
        #v3 = AND(v1, v2)
        #v4 = NOT(v3)
        #v5 = AND(element[2], element[3])
        #value = AND(v4, v5)
        F.append(int(value))
    print(F)
    return F


def unit_step(x): return 0 if x < 0 else 1


def nnm_BF(X, W, F):
    # F - outputs
    eta = 0.3  # eductation norma
    k = 0
    iterations_number = 25
    W = np.array(W)
    for element in X:
        element.insert(0, 1)  # fictional variable x0 added

    training_data = []
    for i in range(0, len(F)):
        training_data.append((np.array(X[i]), F[i]))

    while k < iterations_number:
        predicitons = []
        errors = []  # np.ones(len(F)) vector for errors (actual - predictions)
        print(f'Era number: {k}')
        print(f'Weights: {W}')
        for i in range(0, len(training_data)):
            result = np.dot(W, training_data[i][0])
            predicitons.append(unit_step(result))

            expected = training_data[i][1]
            error = expected - unit_step(result)
            errors.append(error)
            W += [eta * error * x for x in X[i]]
        print(f'Output vector y: {predicitons}')
        print(f'Sum errors: {np.sum(errors)}')
        k += 1
    return


if __name__ == "__main__":
    my_task = initialize()
    nnm_BF(X=my_task[0], W=my_task[1], F=my_task[2])
