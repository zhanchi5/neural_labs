from operator import and_, or_, not_
import numpy as np
import pdb


class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.3):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights)
        pdb.set_trace()
        if summation > 0:
            activation = 1.
        else:
            activation = 0.
        return activation

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


def generate_inputs(number_of_variables):
    X = []
    i = 0
    while i < 2**int(number_of_variables):
        x = list(format(i, f'0{number_of_variables}b'))
        x = [int(n) for n in x]
        X.append(x)
        i += 1
    return X


def get_F(X):
    F = list()
    target_function = []
    for element in X:
        # v1 = OR(element[0], element[1])
        # v2 = AND(v1, element[2])
        # v3 = AND(element[2], element[3])
        # v4 = OR(v2, v3)
        # value = NOT(v4)  # - - my function
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
    for i in range(len(X)):
        F.append((np.array(X[i]), target_function[i]))
    return F


def AND(x1, x2):
    return and_(x1, x2)


def OR(x1, x2):
    return or_(x1, x2)


def NOT(x):
    return not_(x)


if __name__ == "__main__":
    training_inputs = generate_inputs(4)
    # pdb.set_trace()
    labels = [x[1] for x in get_F(training_inputs)]

    perceptron = Perceptron(4)
    perceptron.train(training_inputs, labels, 50)
    print(perceptron.weights)
