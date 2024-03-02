'''
Some maybe useful functions, taken from https://github.com/gkdziugaite/pacbayes-opt/
'''

import numpy as np
import torch
from math import ceil, floor

def quantize_lambda(rho, device, b=100, c=0.1):
    lbda = torch.exp(2*rho).item()
    j = b * np.log(c / lbda); j_plus = ceil(j); j_minus = floor(j)
    lbda_plus = c * np.exp(-j_plus / b); lbda_minus = c * np.exp(-j_minus / b)
    rho_plus = 0.5 * np.log(lbda_plus); rho_minus = 0.5 * np.log(lbda_minus)
    rho_plus = torch.FloatTensor(rho_plus); rho_minus = torch.FloatTensor(rho_minus)
    
    return rho_plus, rho_minus

def bound_objective(model, loader, scorer, w, w0, sigma, rho, d, m, device, delta=0.025, b=100, c=0.1):
    
    loss_term = loss(model, loader, scorer, device)
    bre_term = torch.sqrt(0.5 * B_RE(w, w0, sigma, rho, d, m, delta, b, c))   ## !!!
    # bre_term = 0
    return loss_term + bre_term

def loss(model, loader, scorer, device) :
    loss = torch.Tensor([0.]).to(device)
    
    for batch in loader:
        x ,y = batch
        loss += scorer.loss(model(x.to(device)), y.to(device))

    return loss  / len(loader)

def B_RE(w, w0, sigma, rho, d, m, delta=0.025, b=100, c=0.1):
    KL = 1/ torch.exp(2*rho) *torch.sum(torch.exp(2*sigma)) - d + 1 / torch.exp(2*rho) * torch.norm(w-w0)**2 ## NEW: I think it is ||w - w0||^2, isn't it?
    KL = KL / 2.0
    KL = KL + d* rho 
    KL = KL -  torch.sum(sigma) 
    B_RE =1/(m-1) * (KL + 2 * torch.log(b*np.log(c) - 2*rho*b )  + np.log( np.pi**2 * m / (6 * delta)))
    return B_RE

def KLdiv(pbar,p):
    return pbar * np.log(pbar/p) + (1-pbar) * np.log((1-pbar)/(1-p))


def KLdiv_prime(pbar,p):
    return (1-pbar)/(1-p) - pbar/p

def Newt(p,q,c):
    newp = p - (KLdiv(q,p) - c)/KLdiv_prime(q,p)
    return newp

def approximate_BPAC_bound(train_accur, B_init, niter=5):
    B_RE = 2* B_init **2
    A = 1-train_accur
    B_next = B_init+A
    print(B_next)
    if B_next>1.0:
        return 1.0
    for i in range(niter):
        B_next = Newt(B_next,A,B_RE)
    return B_next


def hoeffdingbnd(M,delta):
    eps = np.sqrt(np.log(2/delta)/M)
    return eps


def SamplesConvBound( train_error=0.028, M=1000, delta=0.01, p_init=None, niter=5):
    c =  np.log(2/delta)/M
    if p_init is None:
        p_init = hoeffdingbnd(M,delta)
        print("Hoeffding's error", p_init)
    p_next = p_init+train_error
    for i in range(niter):
        p_next = Newt(p_next,train_error,c)
    print("Chernoff's error", p_next-train_error)
    return p_next-train_error
