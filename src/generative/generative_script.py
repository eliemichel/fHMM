import numpy as np

M=2
K=3
T=10
N=100

pi_l=list()
pi_l.insert(0,np.array([1/3,1/3,1/3]))
pi_l.insert(1,np.array([0,1,0]))

A1=np.array([[1/2, 1/2, 0],[1/3, 1/3, 1/3],[1, 0, 0]])
A2=np.array([[1/3, 1/3, 1/3],[1/2, 1/2, 0],[0, 1, 0]])
A_l=list()
A_l.insert(0,A1)
A_l.insert(1,A2)

C_m=np.array([[1,1/2],[1/2,1]])

W_l=list()
W1=np.array([[-2,-2,-1],[-2,-1,-2]])
W1=W1.astype(float)
W2=-W1
W_l.insert(0,W1)
W_l.insert(1,W2)

X_l=list()
Y_l=list()

for k in np.arange(N):
    Xk,Yk=generative_fun(pi_l,A_l,C_m,W_l,T)
    X_l.insert(k,Xk)
    Y_l.insert(k,Yk)