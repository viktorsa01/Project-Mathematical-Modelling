import numpy as np

a = 0; b = 1
M = 10

x = np.linspace(a, b, M+1)
h = (b-a)/M
kappa = lambda x: 1-x**2


def tridiag(sub, mid, sup, N):

    e = np.ones(N)

    A = sub*np.diag(e[1:], -1) 
    A += mid*np.diag(e) 
    A += sup*np.diag(e[1:], 1)

    return A

def getDiags(kappa, x):
    
    mod_x = x + h/2

    sub = kappa(mod_x[1:-1])
    sup = kappa(mod_x[0:-2])
    mid = - (sub + sup)

    return sub, mid, sup












    

