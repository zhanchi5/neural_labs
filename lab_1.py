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
        # value = NOT(v4)
        v1 = element[0]
        v2 = element[1]
        v3 = AND(v1, v2)
        v4 = NOT(v3)
        v5 = AND(element[2], element[3])
        value = AND(v4, v5)
        F.append(int(value))
    return F


def unit_step(x): return 0 if x < 0 else 1


def nnm_BF(X, W, F):
    # F - outputs
    eta = 0.3  # eductation norma
    k = 0
    iterations_number = 25
    for element in X:
        element.insert(0, 1)
    # Initializing additional parameters to compute SSE
    predictions = np.ones(len(F))     # vector for predictions

    errors = []  # vector for errors (actual - predictions)

    while k < iterations_number:
        print(f'Era number: {k}')
        for i in range(0, len(X)):
            result = np.dot(W, X[i])
            error = F[i] - unit_step(result)
            errors.append(error)
            print(eta)
            print(error)
            print(X[i])
            W += eta * error * X[i]
        k += 1
    pdb.set_trace()
            # summation step
#            f = np.dot(X[i], W)
            # activation function

#            if f > eta:
#                predictions[i] = 1.
#            else:
#                predictions[i] = 0.

            # updaiting weights
#            for j in range(0, len(W)):
#                W[j] = W[j] + eta * (F[i]-predictions[i]) * X[i][j]
#        pdb.set_trace()

       # for i in range(0, len(F)):
       #     errors[i] = F[i]-predictions[i]
       # print(np.sum(errors))
        # sum_error = sum(errors-yhat_vec)
#        print(f'Summary error E: {sum(errors - yhat_vec)}')
#        for i in range(0, len(F)):
#            errors[i] = (F[i] - yhat_vec[i])**2
#        pdb.set_trace()

    return W


# def plot_data():
#    pass


if __name__ == "__main__":
    my_task = initialize()
    weights = nnm_BF(X=my_task[0], W=my_task[1], F=my_task[2])
    # print(f"The weights are: {weights}", "\n")
    # print(f'Summary of errors: {summary_of_errors}', "\n")
