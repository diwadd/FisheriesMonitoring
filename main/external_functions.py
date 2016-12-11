import math
import numpy as np


def logloss(y, p):

    print("y len: " + str(len(y)))
    print("p len: " + str(len(p)))

    print("y[0]:  " + str(y[0]))
    print("p[0]:  " + str(p[0]))

    print("y[1]:  " + str(y[1]))
    print("p[1]:  " + str(p[1]))

    N = len(y)
    ll = 0.0
    for i in range(N):
        ll = ll + np.dot(y[i], np.log( np.maximum( np.minimum(p[i], 1 - np.power(10.0, -15.0)), np.power(10.0, -15.0) ) ))
    ll = -ll/N

    return ll



#y1 = [[0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]]
#p2 = [[-0.43280941,  0.07222891,  1.3835988,  -0.43280941, -0.43280941, -0.45246649, -0.45246649, -0.45246649],
#      [-0.43280941, 2.07222891, 1.3835988, -0.43280941, -0.43280941, 2.45246649, -1.45246649, -1.45246649]]


#print(logloss(y1, p2))