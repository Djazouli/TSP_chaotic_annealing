import numpy as np
import math
import matplotlib.pyplot as plt
import time
import tsplib


def g(x, epsilon):
    y = 0.5*(1. + math.tanh(x/epsilon))
    return y


class ChaoticHopfieldNetwork:

    def __init__(self, W1, W2, k, I0, distMatrix, alpha, beta, epsilon, z0, itersMax):
        self.size, nothing = distMatrix.shape
        self.W1 = W1
        self.W2 = W2
        self.k = k
        self.I0 = I0
        self.distMatrix = distMatrix
        self.normDistMatrix = distMatrix / distMatrix.max()
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = beta
        u00 = epsilon * np.arctanh(np.array([2. / self.size - 1]))[0]
        N = np.random.uniform(-0.1 * math.fabs(u00), 0.1 * math.fabs(u00), [self.size, self.size])
        self.X = u00 + N  # X = activation
        self.Y = np.zeros((self.size, self.size))  # Y = output
        self.rangeSize = range(self.size)
        self.z = z0
        self.pairs = []
        for i in range(self.size):
            for j in range(self.size):
                self.pairs.append((i, j))
        np.random.shuffle(self.pairs)
        self.iters = 0
        self.itersMax = itersMax
        self.converged = False

    # def assignInitialState(self):
    #     size = self.size
    #     epsilon = self.epsilon
    #     u00 = epsilon * np.arctanh(np.array([2. / size - 1]))[0]
    #     N = np.random.uniform(-0.1 * math.fabs(u00), 0.1 * math.fabs(u00), [size, size])
    #     self.X = u00 + N

    # def assignInitialState(self):
    #     size = self.size
    #     self.X += np.random.uniform(-1,1,(size,size))

    def update(self):
        for k, j in self.pairs:
            self.updateNeuron(k, j)
        self.updateZ()
        self.iters += 1
        self.converged = self.iters > self.itersMax or self.valid_tour()

    def updateNeuron(self,i,k): # Update the neuron i,k
        n, X, Y = self.size, self.X, self.Y
        W1, W2, alpha = self.W1, self.W2, self.alpha
        ds = self.normDistMatrix
        rangeSize = self.rangeSize
        a = -W1 * (
            sum(Y[i, l] if l != k else 0.0 for l in rangeSize) +
            sum(Y[j, k] if j != i else 0.0 for j in rangeSize)
        )
        b = -W2 * sum(ds[i, j] * (Y[j, (k + 1) % n] + Y[j, (k - 1) % n]) if j != i else 0.0 for j in rangeSize)

        c = self.k * X[i, k] - self.z * (Y[i,k] - self.I0)

        self.X[i, k] = alpha * (a + b + W1) + c
        self.Y[i, k] = g(self.X[i, k], self.epsilon)

    def valid_rows(self):
        return [len(np.where(self.Y[i])[0]) == 1 for i in self.rangeSize]

    def valid_cols(self):
        YT = self.Y.transpose()
        return [len(np.where(YT[i])[0]) == 1 for i in self.rangeSize]

    def n_valid_rows(self):
        return len(np.where(self.valid_rows())[0])

    def n_valid_cols(self):
        return len(np.where(self.valid_cols())[0])

    def percent_valid(self):
        total = self.n_valid_rows() + self.n_valid_cols()
        return total / (2.0 * self.size)

    def valid_tour(self):
        return self.percent_valid() == 1

    def updateZ(self):
        self.z = (1 - self.beta)*self.z


def extractRoute(solveResult):
    size, nothing = solveResult.shape
    route = []
    for j in range(size):
        found = False
        for i in range(size):
            if solveResult[i,j] > 0.7:
                if found:
                    return [], False
                found = True
                route.append(i)
            if i == size - 1 and not found:
                return [], False
    return route, True


def calcLength(route,distMatrix):
    numberOfCities, qq = distMatrix.shape
    lengthTot = 0
    for i in range(1,numberOfCities):
        lengthTot = lengthTot + distMatrix[route[i-1],route[i]]
    lengthTot = lengthTot + distMatrix[route[numberOfCities-1],route[0]]
    return lengthTot


def solve(distMatrix, W1, W2, k, I0, alpha, beta, epsilon, z0, itersMax):
    network = ChaoticHopfieldNetwork(W1, W2, k, I0, distMatrix, alpha, beta, epsilon, z0, itersMax)
    while not network.converged:
        network.update()
    return network.Y


def represent(numberOfTry, distMatrix, W1, W2, k, I0, alpha, beta, epsilon, z0, itersMax):
    start_time = time.time()
    result = []
    for i in range(numberOfTry):
        print(i)
        x, correct = extractRoute(solve(distMatrix, W1, W2, k, I0, alpha, beta, epsilon, z0, itersMax))
        if correct:
            nb = calcLength(x, distMatrix)
        else:
            nb = -1
        result.append(nb)
    print("--- Chaotic Hopfield : %s seconds ---" % (time.time() - start_time))
    print(result)
    yo = result.copy()
    try:
        yo.remove(-1.)
    except:
        print('No fail')
    print(min(yo))
    plt.hist(result, histtype='bar', align='mid', rwidth=0.5, bins=range(int(min(result)-1), int(max(result))+1))
    plt.show()


matrix = tsplib.distance_matrix('gr17.xml')
print(matrix)

represent(10, matrix, 1, 1, 0.9, 0.5, 0.015, 0.01, 0.004, 0.1, 500)