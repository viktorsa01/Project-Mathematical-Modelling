# %%

import numpy as np
import matplotlib.pyplot as plt

# Define parameters and functions
D = 0.3  # Diffusion coefficient
B = 1.45  # Reaction coefficient

# Define the range of x and number of points


D = 0.3
B_out = 1.45
A_out = 201.4
S_2 = -0.477
x_s = 0.95
a_u = 0.38
a_l = 0.68

x_start, x_end = 0, 1
M = 1000
x = np.linspace(x_start, x_end, M)

dx = x[1] - x[0]

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
    return (A_out) / (B_out*(alpha_0_hat(x_s)*y(x_s) + a_l/B_out + S_2*a_l*P_2(x_s)/(6*D+B_out)))


def a(x,x_s):
    a_arr = np.zeros(len(x))
    for i in range(len(a_arr)):
        if x[i] <= x_s:
            a_arr[i] = a_l
        elif x[i] >= x_s:
            a_arr[i] = a_u
    return a_arr

S = lambda x: 1+S_2*0.5*(3*x**2-1)

# Define K(x) and f(x)
def K(x):
    return 1 - x**2

# Central difference method
K_values = K(x)  # Precompute K(x)
f_values = -A_out + Q(x_s)*S(x)*a(x,x_s)

# Construct the system of equations
A = np.zeros((M, M))
b = np.zeros(M)
for i in range(1, M-1):
    # Compute coefficients

    A[i, i - 1] = -D * K(x[i]) / dx**2 - D*x[i]/dx 
    A[i, i] = D * (2*K(x[i])) / dx**2 + B
    A[i, i + 1] = -D * K(x[i]) / dx**2 +  D*x[i]/dx
    b[i] = f_values[i]


A[0, 0] = 1
A[0, 1] = -1
b[0] = 0

A[-1, -1] = 1
A[-1,-2] = -1
b[-1] = 0

P = int(M*x_s)

# Impose smooth funciton at x_s 
A[P,P] = 1
A[P,P-1] = 1
A[P,P+1] = 1
b[P] = 0


# Solve the system of equations
T = np.linalg.solve(A, b)

# Plot the solution
plt.plot(x, T, label="T(x)")
plt.xlabel("x")
plt.ylabel("T(x)")
plt.title("Solution")
plt.legend()
plt.grid()
plt.show()

# %%
