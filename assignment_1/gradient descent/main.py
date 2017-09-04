import random
import matplotlib.pyplot as plt
import json


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


if __name__ == "__main__":
    data = createData()
    apples, oranges = sortData(data)
    drawScatterPlot(apples, oranges)
