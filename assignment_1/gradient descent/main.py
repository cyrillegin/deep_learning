import matplotlib.pyplot as plt
import numpy as np
from gradient_descent import takeStep, setup, getZ


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
    for i in range(0, 5):
        x = np.linspace(-2, 2, 100)
        plt.scatter(oranges[0], oranges[1], c='b')
        plt.scatter(apples[0], apples[1], c='r')
        stepCounter = takeStep(stepCounter)
        print stepCounter['weights']
        plt.plot(x, (-stepCounter["weights"][0] / stepCounter['weights'][1])*x, c='g')
        plt.show()


def f(x):
    return 0.5*x[0]**2 + 2.5*x[1]**2


def createContours(stepCounter):
    xmesh, ymesh = np.mgrid[-2:2:10j, -2:2:10j]
    fmesh = getZArr(stepCounter)
    print "got bak"
    print fmesh
    # print xmesh
    return fmesh


def getZArr(stepCounter):
    coords = []
    for i in range(0, 5):
        for j in range(0, 5):
            coords.append((i, j))
    stepCounter['inputs'] = coords
    print "sending:"
    print coords
    return getZ(stepCounter)


def drawContourLine(contours):

    plt.axis("equal")
    plt.contour(contours[0], contours[1], contours[2])
    plt.show()


if __name__ == "__main__":
    stepCounter = setup()
    data = createData()
    apples, oranges = sortData(data)
    # drawScatterPlot(apples, oranges, stepCounter)
    contours = createContours(stepCounter)
    drawContourLine(contours)
