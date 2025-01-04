import numpy as np
# data is a list of peak tuples -> (freq, time)
def generate_tuples(data: np.ndarray):
    lis = []
    for each in range(len(data)):
        l = []
        for i in range(1, 16):
            if ((each + i) <= (len(data) - 1)):
                t = (data[each][0], data[each + i][0], data[each + i][1] - data[each][1]), data[each][1]
                l.append(t)

        lis.append(l)

    return lis