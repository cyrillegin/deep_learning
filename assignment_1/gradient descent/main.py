import matplotlib.pyplot as plt
import numpy as np
from gradient_descent import takeStep, setup


def createData():
    data = []
    with open('fruit_data.txt', 'r') as txt:
        for i in txt.readlines():
            data.append(i.split())
    return data


def sortData(data):
    apples = [[], []]
    oranges = [[], []]
    for i in data:
        if i[2] == 1:
            apples[0].append(i[0])
            apples[1].append(i[1])
        else:
            oranges[0].append(i[0])
            oranges[1].append(i[1])
    return apples, oranges


def drawScatterPlot(apples, oranges, stepCounter):
    # Only display the first five steps.
    for i in range(0, 5):
        x = np.linspace(-2, 2, 100)
        plt.scatter(oranges[0], oranges[1], c='b')
        plt.scatter(apples[0], apples[1], c='r')
        stepCounter = takeStep(stepCounter)
        plt.plot(x, (-stepCounter["weights"][0] / stepCounter['weights'][1])*x, c='g')
        plt.show()


def f(x):
    return 0.5*x[0]**2 + 2.5*x[1]**2


def drawContourLine(stepCounter):
    A = np.array([stepCounter['deltaWeights'][0], stepCounter['deltaWeights'][1]])
    b = np.array([stepCounter['finalOutput'][0], stepCounter['finalOutput'][1], ])
    x = np.ones((np.shape(A)[0], 1))

    x_ini = -2
    x_end = 2
    y_ini = -2
    y_end = 2
    rate_of_points = 0.25

    x1, x2 = np.meshgrid(np.arange(x_ini, x_end, rate_of_points), np.arange(y_ini, y_end, rate_of_points))
    fq = np.zeros(np.shape(x1))

    for i in xrange(len(x1)):
        for j in xrange(len(x2)):
            x = np.array([x1[i][j], x2[i][j]])
            fq[i][j] = np.dot(x, np.dot(A, x)) - 2 * np.dot(x, b)

    plt.figure()
    plt.contour(x1, x2, fq, colors='k')

    plt.show()
    plt.close()


if __name__ == "__main__":
    stepCounter = setup()
    data = createData()
    apples, oranges = sortData(data)
    drawScatterPlot(apples, oranges, stepCounter)
    stepCounter = setup()
    drawContourLine(stepCounter)
