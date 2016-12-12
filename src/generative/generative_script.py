from generative_fun import generative_fun
import numpy as np

M = 2
K = 3
T = 10
N = 100

pi_l = [
    np.array([1/3,1/3,1/3]),
    np.array([0,1,0]),
]

A1 = np.array([[1/2, 1/2, 0],[1/3, 1/3, 1/3],[1, 0, 0]])
A2 = np.array([[1/3, 1/3, 1/3],[1/2, 1/2, 0],[0, 1, 0]])
A_l = [A1, A2]

C_m = np.array([[1,1/2],[1/2,1]])

W1 = np.array([[-2,-2,-1],[-2,-1,-2]])
W1 = W1.astype(float)
W_l = [W1, -W1]

X_l = []
Y_l = []

for k in range(N):
    Xk, Yk = generative_fun(pi_l, A_l, C_m, W_l, T)
    X_l.append(Xk)
    Y_l.append(Yk)
