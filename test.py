# %%

import numpy as np
import matplotlib.pyplot as plt

D = 0.3  

B_out = 1.45
A_out = 201.4

S_2 = -0.477
x_s = 0.95
a_u = 0.38
a_l = 0.68
tolerance = 1e-10

x_start, x_end = 0, 1
M = 500
x = np.linspace(x_start, x_end, M)

h = x[1] - x[0]


def P_2(x):
    return 1/2*(3*x**2-1)

def y(x):
    alpha_old = 1.0
    alpha_new = 0.0
    res = 1.0 #The 1 represents alpha_0
    for n in range (1000):
        alpha_new = ((2*n*(2*n+1)+B_out/D) / ((2*n+2)*(2*n+1))) *alpha_old
        res += alpha_new * x**(2*n+2)
        if abs(alpha_new) < tolerance:  # Convergence check
            break
        alpha_old = alpha_new
    return res

def y_prime(x):
    alpha_old = 1.0
    alpha_new = 0.0
    res = 0.0 #Alpha_0 is no longer part of the sum due to derivation 
    for n in range (1000):
        alpha_new = ((2*n*(2*n+1)+B_out/D) / ((2*n+2)*(2*n+1))) *alpha_old
        res += (2*n+2)*alpha_new * x**(2*n+1)
        if abs(alpha_new) < tolerance:  # Convergence check
            break
        alpha_old = alpha_new
    return res

def z(s):
    res = 1 + B_out/(2*D) * s + (2+B_out/D)/8 * s**2
    beta_old = (2+B_out/D)/8
    beta_new = 0.0
    for n in range (2, 1000):
        beta_new = ((n*(n+1) +B_out/D) / (2*(n+1)**2)) * beta_old
        res += beta_new * s**(n+1)
        beta_old = beta_new
    return res

def z_prime(s):
    res = B_out/(2*D)  + (2+B_out/D)/8 * 2*s
    beta_old = (2+B_out/D)/8
    beta_new = 0.0
    for n in range (2, 1000):
        beta_new = ((n*(n+1) +B_out/D) / (2*(n+1)**2)) * beta_old
        res += beta_new * (n+1) * s**(n)
        beta_old = beta_new
    return -res 

def alpha_0_hat(x_s):
    gamma = ( (S_2*P_2(x_s)) / (6*D + B_out) + 1/B_out) * (a_u-a_l)
    lambd = (S_2*3*x_s / (6*D + B_out)) * (a_u-a_l)
    return ( z_prime(1-x_s)*gamma - z(1-x_s)*lambd ) / ( y(x_s)*z_prime(1-x_s) - y_prime(x_s)*z_prime(1-x_s) )

def beta_0_hat(x_s, alpha_0_hat):
    gamma = ( (S_2*P_2(x_s)) / (6*D + B_out) + 1/B_out) * (a_u-a_l)
    return ((alpha_0_hat*y(x_s)-gamma))/z(1-x_s)

def Q(x_s):
    return (A_out) / (B_out*(alpha_0_hat(x_s)*y(x_s) + a_l/B_out + S_2*a_l*P_2(x_s)/(6*D+B_out)))

def a(x,x_s):
    return (x<=x_s)*a_l + (x>x_s)*a_u

def T_p(x, x_s):
    return (Q(x_s)*a(x,x_s)-A_out)/B_out + Q(x_s)*S_2*P_2(x)*a(x,x_s)/(6*D+B_out)

def T_p_over(x, x_s):
    return (Q(x_s)*a_l-A_out)/B_out + Q(x_s)*S_2*P_2(x)*a_l/(6*D+B_out)

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
    A[i, i - 1] = -D * K(x[i]) / h**2 - D*x[i]/h 
    A[i, i] = D * (2*K(x[i])) / h**2 + B_out
    A[i, i + 1] = -D * K(x[i]) / h**2 +  D*x[i]/h
    b[i] = f_values[i]



# A[0, 0] = 2*D*K(x[0])/h**2 + B_out
# A[0, 1] = -2*D*K(x[0])/h**2
# b[0] = f_values[0]

# A[-1, -1] = 2*D*K(x[-1])/h**2 + B_out
# A[-1,-2] = -2*D*K(x[-1])/h**2
# b[-1] = f_values[-1]

A[0, 0] = -1
A[0, 1] = 1
b[0] = 0

A[-1, -1] = 1
A[-1,-2] = -1
b[-1] = 0

P = int(M*x_s)

# Impose continuous funciton at x_s 
A[P,P] = 1
A[P,P-1] = 1
A[P,P+1] = 1
b[P] = 0

# Solve the system of equations
T = np.linalg.solve(A, b)



# Plot the solution
plt.plot(x, T,label = "$T$")
plt.xlabel("$x$",fontsize = 16)
plt.ylabel("$T$ [$\degree $C]",fontsize = 16)
plt.axvline(x=x_s, color="orange", linestyle="--", label=f"$x_s = {x_s}$",linewidth = 1)
plt.legend()
plt.grid()
plt.show()


x_s_arr = np.copy(x)

Q_arr = Q(x_s_arr)
plt.plot(x_s_arr[100:], Q_arr[100:])
plt.xlabel("$x_s$")
plt.ylabel("$Q$")
plt.grid()
plt.show()

Q_arr = Q(x_s_arr)
plt.plot(x_s_arr, Q_arr)
plt.xlabel("$x_s$")
plt.ylabel("$Q$")
plt.grid()
plt.show()


x_under_x_s = np.linspace(0,x_s, int(M*x_s))
x_over_x_s = np.linspace(x_s,1, int(M - M*x_s))

a_0 = alpha_0_hat(x_s)
b_0 = beta_0_hat(x_s, a_0)

sol_under_x_s = Q(x_s)*a_0* y(x_under_x_s) + T_p(x_under_x_s, x_s)
sol_over_x_s = Q(x_s)*b_0*z(1-x_over_x_s) + T_p(x_over_x_s, x_s)


plt.plot(x_under_x_s, sol_under_x_s, label = "Under $x_s$")
plt.plot(x_over_x_s[1:], (sol_over_x_s[1:]), label  = "Over $x_s$")
plt.xlabel("$x$",fontsize = 16)
plt.ylabel("$T$ [$\degree $C]",fontsize = 16)
plt.legend()
plt.grid()
plt.show()
x_anal = np.concatenate((sol_under_x_s,sol_over_x_s[1:]))
T = np.delete(T,P+1)
x = np.delete(x,P+1)

plt.plot(x,abs(x_anal-T))
plt.xlabel("$x$",fontsize = 16)
plt.ylabel("Error [$\degree $C]",fontsize = 16)
plt.title("")
plt.grid()
plt.show()


# %%
