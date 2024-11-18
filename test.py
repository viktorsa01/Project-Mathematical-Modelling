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
x_s = 0.9465 # x_s som passer til 1360/4
a_u = 0.38
a_l = 0.68

x_start, x_end = 0, 1
M = 1000
x = np.linspace(x_start, x_end, M)

dx = x[1] - x[0]

def a(x,x_s):
    a_arr = np.zeros(len(x))
    for i in range(len(a_arr)):
        if x[i] <= x_s:
            a_arr[i] = a_l
        elif x[i] >= x_s:
            a_arr[i] = a_u
    return a_arr
def Q(x_s):
    return 1360/4

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
