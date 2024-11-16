# %%
import numpy as np
import matplotlib.pyplot as plt
a = 0; b = 1
M = 100
D = 0.3
B_out = 1.45
A_out = 201.4
S_2 = -0.477
x_s = 0.1
a_u = 0.38
a_l = 0.68


x = np.linspace(a, b, M+1)
h = (b-a)/M
kappa = lambda x: 1-x**2
S = lambda x: 1+S_2*0.5*(3*x**2-1)

def P_2(x):
    return 1/2*(3*x**2-1)

def y(x):
    sum = 1
    for n in range (100):
        sum += ( (2*n*(2*n+1)+B_out/D) / (2*n+2)*(2*n+1) )**(n+1) * x**(2*n+2)
    return sum

def y_prime(x):
    sum = 0
    for n in range (100):
        sum += ( (2*n*(2*n+1)+B_out/D) / (2*n+2)*(2*n+1) )**(n+1) * (2*n+2)*x**(2*n+1)
    return y_prime

def z(x):
    #Må bli enig med viktor om utrykket
    sum = 0
    for n in range (100):
        sum += 0
    return sum

def z_prime(x):
    #Må bli enig med viktor om utrykket
    sum = 0
    for n in range (100):
        sum += 0
    return sum

def alpha_0_hat(x_s):
    gamma = ( (S_2*P_2(x_s)) / (6*D + B_out) + 1/B_out) * (a_u-a_l)
    lambd = (S_2*3*x_s / (6*D + B_out)) * (a_u-a_l)
    return ( z_prime(x_s)*gamma - z(x_s)*lambd ) / ( y(x_s)*z_prime(x_s) - y_prime(x_s)*z_prime(x_s) )

def Q(x_s):
    #Legg inn Q
    return (A_out + 273.15) / (B_out*(alpha_0_hat(x_s)*y(x_s) + a_l/B_out + S_2*a_l*P_2(x_s)/(6*D+B_out)))

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
    A[0,0:2] = 2*kappa_half
    kappa_M_half = kappa(x[M]-h/2)
    A[M,M-1:M+1] = 2*kappa_M_half

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
print(A)
BT = np.eye(M+1)*B_out

LHS = -D*A + BT
RHS = np.ones(M+1)*-A_out + Q(x_s)*S(x)*a(x,x_s)

T = np.linalg.solve(LHS,RHS)

plt.plot(x,T)

# %%
