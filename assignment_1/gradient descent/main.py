import random


def createData():
    data = []
    for i in range(0, 100):
        data.append([int(random.random() * 100), int(random.random() * 100)])
    return data


if __name__ == "__main__":
    i = createData()
    print i
