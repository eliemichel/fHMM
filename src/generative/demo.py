import numpy as np
from numpy import *
from generative import generative
from exact_inference import exact_inference, exact_inference_E
from gibbs_sampler import gibbs_sampler, gibbs_sampler_E

"""
Generate hidden variables and observations to be toy examples for our
implementation.
"""

M=2
K=3
T=10
N=100

pi_l = [np.array([1/3, 1/3, 1/3]), np.array([1/8, 1/8, 3/4])]

A_l = ([np.array([[1/2, 1/4, 1/4],[1/3, 1/3, 1/3],[1/4, 1/4, 1/2]]), 
    np.array([[1/3, 1/3, 1/3],[1/2, 1/4, 1/4],[1/6, 1/6, 2/3]])])

C_m = np.array([[1, 1/2], [1/2, 1]])

W_l = []
W1 = np.array([[-2, -2, -1], [-2, -1, -2]])
W1 = W1.astype(float)
W2 = -W1
W_l.append(W1)
W_l.append(W2)

S_l = []
Y_l = []

for k in range(N):
    [Sk_m, Yk_m] = generative(pi_l, A_l, C_m, W_l, T)
    S_l.append(Sk_m)
    Y_l.append(Yk_m)
    
"""
Compute the E-step using the exact inference.
"""

logpi_l = log(pi_l)
logA_l = log(A_l)
Y_m = Yk_m

[alpha_l, beta_l, gamma_l] = exact_inference(logpi_l, logA_l, C_m, W_l, Y_m)
[E_l, EmixM_l, EmixT_l] = \
exact_inference_E(alpha_l, beta_l, gamma_l, logA_l, C_m, W_l, Y_m)


"""
Compute the E-step using the Gibbs sampler.
"""

Ns = 50
[S_l] = gibbs_sampler(logpi_l, logA_l, C_m, W_l, Y_m, Ns)
[Eg_l, EmixMg_l, EmixTg_l] = gibbs_sampler_E(S_l, K)