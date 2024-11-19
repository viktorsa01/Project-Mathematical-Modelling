# %%
import numpy as np
import matplotlib.pyplot as plt
a = 0; b = 1
M = 100
D = 0.3
B_out = 1.45
A_out = 201.4
S_2 = -0.477
x_s = 0.95
a_u = 0.38
a_l = 0.68


x = np.linspace(a, b, M+1)
h = (b-a)/M
kappa = lambda x: 1-x**2
S = lambda x: 1+S_2*0.5*(3*x**2-1)

def P_2(x):
    return 1/2*(3*x**2-1)

def y(x):
    alpha_old = 1
    alpha_new = 0
    sum = 1 #The 1 represents alpha_0
    for n in range (1000):
        alpha_new = (2*n*(2*n+1)+B_out/D) / ((2*n+2)*(2*n+1)) *alpha_old
        sum += alpha_new * x**(2*n+2)
        alpha_old = alpha_new
    return sum

def y_prime(x):
    alpha_old = 1
    alpha_new = 0
    sum = 0 #Alpha_0 is no longer part of the sum due to derivation 
    for n in range (1000):
        alpha_new = (2*n*(2*n+1)+B_out/D) / ((2*n+2)*(2*n+1))*alpha_old
        sum += (2*n+2)*alpha_new * x**(2*n+1)
        alpha_old = alpha_new
    return sum

def z(s):
    sum = 1 - B_out/(2*D) * s + (2+B_out/D)/8 * s**2
    beta_old = (2+B_out/D)/8
    beta_new = 0
    for n in range (2, 1000):
        beta_new = (n*(n+1) +B_out/D) / (2*(n+1)**2) * beta_old
        sum += beta_new * s**(n+1)
        beta_old = beta_new
    return sum

def z_prime(s):
    sum = - B_out/(2*D)  + (2+B_out/D)/8 * 2*s
    beta_old = (2+B_out/D)/8
    beta_new = 0
    for n in range (2, 1000):
        beta_new = (n*(n+1) +B_out/D) / (2*(n+1)**2) * beta_old
        sum += beta_new * (n+1) * s**(n)
        beta_old = beta_new
    return -sum 

def alpha_0_hat(x_s):
    gamma = ( (S_2*P_2(x_s)) / (6*D + B_out) + 1/B_out) * (a_u-a_l)
    lambd = (S_2*3*x_s / (6*D + B_out)) * (a_u-a_l)
    return ( z_prime(1-x_s)*gamma - z(1-x_s)*lambd ) / ( y(x_s)*z_prime(1-x_s) - y_prime(x_s)*z_prime(1-x_s) )

def Q(x_s):
    # return (A_out) / (B_out*(alpha_0_hat(x_s)*y(x_s) + a_l/B_out + S_2*a_l*P_2(x_s)/(6*D+B_out)))
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

def applyNeumann(A_temp, x, kappa):
    h = x[1]-x[0]
    M = len(x)-1
     # Apply Neumann
    A = np.zeros((x.size,x.size))
    A[1:-1,1:-1] = A_temp
    
    kappa_half = kappa(x[0]+h/2)
    A[0, 0] = -2*kappa_half
    A[0, 1] =  2*kappa_half
    kappa_M_half = kappa(x[M]-h/2)

    A[M,M-1] = 2*kappa_M_half
    A[M, M] = -2*kappa_M_half

    A[1, 0] = kappa_half
    A[M-1, M] = kappa_M_half


    P = int(x_s*M)
    A[P,P] = 1
    A[P,P-1] = 1
    A[P,P+1] = 1
    

    return A


def getDiags(kappa, x):
    
    mod_x = x + h/2

    sub = kappa(mod_x[1:-1])
    sup = kappa(mod_x[0:-2])
    mid = - (sub + sup)

    return sub, mid, sup

sub, mid, sup = getDiags(kappa,x)
A = tridiag(sub,mid,sup,M-1)

A = applyNeumann(A,x,kappa)

BT = np.eye(M+1)*B_out

LHS = -D*A + BT
RHS = np.ones(M+1)*-A_out + Q(x_s)*S(x)*a(x,x_s)
RHS[int(x_s*M)] = 0

T = np.linalg.solve(LHS,RHS)

plt.plot(x,T)
plt.show()

# %%
 