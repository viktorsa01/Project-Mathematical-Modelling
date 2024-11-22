import numpy as np
import matplotlib.pyplot as plt

def delta(d):
    """
    Returns the declination for a given day d of the year, where d=1 is January 1st
    """
    return np.radians(23.45) * np.sin(np.radians(360 / 365 * (d - 81)))

def mu_d(phi, d):
    """
    Returns the diurnal mean of the cosine of the solar zenith angle for a given latitude phi and day d
    """
    delta_ = delta(d)
    val = - np.tan(phi) * np.tan(delta_)

    if val >= 1:
        h0 = 0
    elif val <= -1:
        h0 = np.pi
    else:
        h0 = np.arccos(val)

    return 4/np.pi * (np.sin(phi)*np.sin(delta_)*h0 + np.cos(phi)*np.cos(delta_)*np.sin(h0))


def mu_a(phi):
    """
    Returns the annual mean of the cosine of the solar zenith angle for a given latitude phi
    """
    integrand = lambda d: mu_d(phi, d)
    return 1/365*np.sum([integrand(d) for d in range(1, 366)])

def S(phi):
    """
    Returns the solar constant for a given latitude phi
    """
    x = np.sin(phi)
    return 1 - 0.477/2 * (3*x**2 - 1)


phi_arr = np.linspace(-np.pi/2, np.pi/2, 100)
x_arr = np.sin(phi_arr)
fit_ = np.polynomial.legendre.Legendre.fit(x_arr, [mu_a(phi) for phi in phi_arr], 2)
print(fit_)
fit_no_tilt = np.polynomial.legendre.Legendre.fit(x_arr, [4/np.pi*np.cos(phi) for phi in phi_arr], 2)
print(fit_no_tilt)

plt.style.use('ggplot')
plt.plot(x_arr, [S(phi) for phi in phi_arr], label='Given S(x)')
plt.plot(x_arr, fit_(x_arr), ls=(0,(5,10)), label='Fit S(x) with tilt')
plt.plot(x_arr, [4/np.pi*np.cos(phi) for phi in phi_arr], label='Fit S(x) without tilt')
plt.xlabel(r'$x=\sin(\phi)$')
plt.ylabel('Intensity')
plt.legend()
plt.show()