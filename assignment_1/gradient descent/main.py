import matplotlib.pyplot as plt
import numpy as np


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


def drawScatterPlot(apples, oranges):
    plt.scatter(oranges[0], oranges[1], c='b')
    plt.scatter(apples[0], apples[1], c='r')
    plt.plot([-2, 2], [-2, 2])
    plt.show()


def f(x):
    return 0.5*x[0]**2 + 2.5*x[1]**2


def createContours():
    xmesh, ymesh = np.mgrid[-2:2:50j, -2:2:50j]
    fmesh = f(np.array([xmesh, ymesh]))
    return [xmesh, ymesh, fmesh]


def getError():
    return


def drawContourLine(contours, errors):

    plt.axis("equal")
    plt.contour(contours[0], contours[1], contours[2])
    plt.show()


if __name__ == "__main__":
    data = createData()
    apples, oranges = sortData(data)
    drawScatterPlot(apples, oranges)
    contours = createContours()
    errors = getError()
    drawContourLine(contours, errors)
