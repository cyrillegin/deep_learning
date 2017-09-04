import random
import matplotlib.pyplot as plt


def createData():
    data = [[], [], [], []]
    for i in range(0, 20):
        data[0].append(int(random.random() * 100))
        data[1].append(int(random.random() * 100))
        data[2].append(int(random.random() * 100))
        data[3].append(int(random.random() * 100))
    return data


def drawScatterPlot(data):
    plt.scatter(data[0], data[1], c='b')
    plt.scatter(data[2], data[3], c='r')
    plt.show()

if __name__ == "__main__":
    data = createData()
    drawScatterPlot(data)
