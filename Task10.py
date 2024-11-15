# %%
import numpy as np
import matplotlib.pyplot as plt
a = 0; b = 1
M = 1000
D = 10
B_out = 1.45
A_out = 200
S_2 = -0.477
x_s = 0.3
a_u = 0.38
a_l = 0.68


x = np.linspace(a, b, M+1)
h = (b-a)/M
kappa = lambda x: 1-x**2
S = lambda x: 1+S_2*0.5*(3*x**2-1)

def Q(x_s):
    #Legg inn Q
    return 1360/4

def a(x,x_s):
    a_arr = np.zeros(len(x))
    for i in range(len(a_arr)):
        if x[i] < x_s:
            a_arr[i] = a_u
        elif x[i] >= x_s:
            a_arr[i] = a_l
    return a_arr

def tridiag(sub, mid, sup, N):
    '''
    Eivind
    '''
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

sub, mid, sup = getDiags(kappa,x)
A = tridiag(sub,mid,sup,M-1)

BT = np.eye(M-1)*B_out

LHS = -D*A + BT
RHS = np.ones(M-1)*-A_out + Q(x_s)*S(x[1:-1])*a(x[1:-1],x_s) 

T = np.linalg.solve(LHS,RHS)

plt.plot(x[1:-1],T)

# %%
