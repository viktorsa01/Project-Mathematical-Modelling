import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def tan_phi(x):
    """
    Returns tangent of phi given x=sin(phi)
    """
    return x/np.sqrt(1+x**2)

@np.vectorize
def h_0(x, delta):
    """
    Returns the hour angle at which sunset occurs
    """
    val = - tan_phi(x) * np.tan(delta)
    if val > 1:
        return np.pi
    elif val < -1:
        return 0
    else:
        return np.arccos(val)
    

def mu_d(x, delta):
    """
    Returns the diurnal mean of the cosine of the solar zenith angle for a given sinus of latitude x and declination delta
    """
    return h_0(x, delta)/np.pi*x*np.sin(delta) + 1/np.pi*np.sqrt(1-x**2)*np.cos(delta)*np.sin(h_0(x, delta))

@np.vectorize
def mu_a(x):
    """
    Returns the annual mean of the cosine of the solar zenith angle for a given a sinus of latitude x
    """
    integrand = lambda delta: mu_d(x, delta)
    delta_0 = np.deg2rad(23.5)
    return 5*sp.integrate.quad(integrand, -delta_0, delta_0)[0]

def S(x):
    return 1 - 0.5*0.477 * (3*x**2 - 1)


x_arr = np.linspace(0, 1, 100)
plt.plot(x_arr, S(x_arr), label='S(x)')
plt.plot(x_arr, mu_a(x_arr), label='mu_a(x)')
plt.xlabel(r'$\sin(\phi)$')
plt.ylabel('Value')
plt.legend()
plt.show()
