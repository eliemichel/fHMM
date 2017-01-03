import numpy as np
import time
from numpy import *
from generative import generative, randinit
from exact_inference import exact_inference, exact_inference_E
from gibbs_sampler import gibbs_sampler, gibbs_sampler_E
from meanfield import meanfield, meanfield_E

"""
Generate hidden variables and observations to be toy examples for our
implementation.
"""

M=3
K=2
T=10
N=20
D = 2

[pi_l, A_l, W_l] = randinit(M, K, D)

C_m = np.array([[1, 0], [0, 1]])

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

Nr = 30
Ns = 10
[S_l] = gibbs_sampler(logpi_l, logA_l, C_m, W_l, Y_m, Nr)
[Eg_l, EmixMg_l, EmixTg_l] = gibbs_sampler_E(S_l, K, Ns)

"""
Compute the E-step using mean-field.
"""

Nit = 10 
[theta_l] = meanfield(logpi_l, logA_l, C_m, W_l, Y_m, Nit)
[Em_l, EmixMm_l, EmixTm_l] = meanfield_E(theta_l)


"""
Compute EM algorithm
"""

N = 20
[pif_l, Af_l, Cf_m, Wf_l, L_l] = EM(pi_l, A_l, C_m, W_l, Y_m, N)

method = 2
[pifg_l, Afg_l, Cfg_m, Wfg_l, Lg_l] = EM(pi_l, A_l, C_m, W_l, Y_m, N, method)

method = 3
[pifm_l, Afm_l, Cfm_m, Wfm_l, Lm_l] = EM(pi_l, A_l, C_m, W_l, Y_m, N, method)

"""
Real test with random initialization
"""
N = 50

[pi0_l, A0_l, W0_l] = randinit(1, K**M, D)
[pif_l, Af_l, Cf_m, Wf_l, L_l] = EM(pi0_l, A0_l, C_m, W0_l, Y_m, N)

[pi0_l, A0_l, W0_l] = randinit(M, K, D)
[pif_l, Af_l, Cf_m, Wf_l, L_l] = EM(pi0_l, A0_l, C_m, W0_l, Y_m, N)

[pi0_l, A0_l, W0_l] = randinit(M, K, D)
method = 2
[pifg_l, Afg_l, Cfg_m, Wfg_l, Lg_l] = EM(pi_l, A_l, C_m, W_l, Y_m, N, method)

[pi0_l, A0_l, W0_l] = randinit(M, K, D)
method = 3
[pifm_l, Afm_l, Cfm_m, Wfm_l, Lm_l] = EM(pi_l, A_l, C_m, W_l, Y_m, N, method)
