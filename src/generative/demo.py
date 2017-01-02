import numpy as np
import time
from numpy import *
from generative import generative, randinit
from exact_inference import exact_inference, exact_inference_E
from gibbs_sampler import gibbs_sampler, gibbs_sampler_E

"""
Generate hidden variables and observations to be toy examples for our
implementation.
"""

M=3
K=2
T=10
N=20

pi_l = [np.array([1/2, 1/2]), np.array([1/3,2/3]), np.array([3/4,1/4])]
#pi_l = [np.array([1/4, 1/3, 5/12])]


A_l = ([np.array([[1/2, 1/2],[1/3, 2/3]]), np.array([[1/2, 1/2],[1/3, 2/3]]), 
                 np.array([[1/2, 1/2],[1/2, 1/2]])])
#A_l = [np.array([[1/2, 1/4, 1/4],[1/3, 1/3, 1/3],[1/4, 1/4, 1/2]])]

C_m = np.array([[1, 0], [0, 1]])

W_l = []
W1 = np.array([[-2, -1], [-2, -2]])
W1 = W1.astype(float)
W2 = -W1
W_l.append(W1)
W_l.append(W2)
W_l.append(W1*3)

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

"""
Compute EM algorithm
"""

N = 20
[pif_l, Af_l, Cf_m, Wf_l, L_l] = EM(pi_l, A_l, C_m, W_l, Y_m, N)

method = 2
Ns = 50
[pifg_l, Afg_l, Cfg_m, Wfg_l, Lg_l] = \
EM(pi_l, A_l, C_m, W_l, Y_m, N, method, Ns)

"""
Real test with random initialization
"""

D = 2
[pi0_l, A0_l, W0_l] = randinit(M, K, D)

N = 20
[pif_l, Af_l, Cf_m, Wf_l, L_l] = EM(pi0_l, A0_l, C_m, W0_l, Y_m, N)

method = 2
Ns = 50
[pifg_l, Afg_l, Cfg_m, Wfg_l, Lg_l] = \
EM(pi_l, A_l, C_m, W_l, Y_m, N, method, Ns)
