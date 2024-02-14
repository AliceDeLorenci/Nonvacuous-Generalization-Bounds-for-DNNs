'''
Some maybe useful functions, taken from https://github.com/gkdziugaite/pacbayes-opt/
'''

import numpy as np
import torch

def KLdiv(pbar,p):
    return pbar * np.log(pbar/p) + (1-pbar) * np.log((1-pbar)/(1-p))


def KLdiv_prime(pbar,p):
    return (1-pbar)/(1-p) - pbar/p

def Newt(p,q,c):
    newp = p - (KLdiv(q,p) - c)/KLdiv_prime(q,p)
    return newp


def approximate_BPAC_bound(train_accur, B_init, niter=5):
    #B_RE = 2* B_init **2
    A = 1-train_accur
    B_next = B_init+A
    if B_next>1.0:
        return 1.0
    for i in range(niter):
        B_next = Newt(B_next,A,B_RE)
    return B_next


def hoeffdingbnd(M,delta):
    eps = np.sqrt(np.log(2/delta)/M)
    return eps


def SamplesConvBound(train_error=0.028,M=1000,delta=0.01,p_init = None, niter = 5):
    c =  np.log(2/delta)/M
    if p_init is None:
        p_init = hoeffdingbnd(M,delta)
        print("Hoeffding's error", p_init)
    p_next = p_init+train_error
    for i in range(niter):
        p_next = Newt(p_next,train_error,c)
    print("Chernoff's error", p_next-train_error)
    return p_next-train_error

def weight_init(size, mu=0, sigma=0.04):
    z = np.random.randn(list(size))
    w = mu + sigma*z
    w = np.ma.masked_where((w > 2*sigma) & (x < -2*sigma ), x)

    return torch.from_numpy(w) 

def get_all_params(model):
    # get all parameters of a torch model as a 1D Tensor

    weights = torch.Tensor()
    for param in model.parameters():
        weights = torch.cat([weights, param.flatten()])

    return weights


