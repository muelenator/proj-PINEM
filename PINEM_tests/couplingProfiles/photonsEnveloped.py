import numpy as np
from scipy.integrate import quad


'''
Coupling profile across all cavities:

'''
env_fn = {
    'name': 'Gaussian',
    'sigma': 1e-6,
    'amp': 1,
    'Pulse width': 5e-12,
    't_delay': 0,
}

# Gaussian envelope
def gauss_env(t, t0, sigma): 
    return np.exp(-(t-t0)**2/2/sigma**2)

# Field experienced by the electron
def fieldExp(Ez, t, t0, z, z0, sigma):
    return Ez(z-z0) * gauss_env(t, t0, sigma)

# Integrate field along the electron trajectory
def 

kz = mode.mode_params[(wg['m'],wg['l'],wg['type'],'kz')]

Ez_ints1 = quad(fieldExp, 0, taus, args=(0, 0, config['Pulse width']))