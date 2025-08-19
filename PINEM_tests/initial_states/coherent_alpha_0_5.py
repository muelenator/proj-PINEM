''' 
This is a two-part state of light. 
The states are unentangled
Each 'part' (functionally, a cavity) is loaded with a coherent state
    Here, alpha = .5
'''

from bkgndLibs.physicalConstants import *
from bkgndLibs.PINEM import coherentStateGenerate
import numpy as np

'''
State of light info:
    type: str
        Type of light state
    alpha1: complex float
        Coherent state parameter in cavity 1 (the first cavity encountered by the electron)
    alpha2: complex float
        Coherent state parameter in cavity 2 (the second cavity encountered by the electron)
    n_states: int
        Number of Fock states to consider
'''
photon_state = {
    'type': 'coherent',
    'alpha1': .5,
    'alpha2': .5,
    'n_states': 15,
    'Not a tensor product': True
}

# Fock space, rho = a|0> + b|1> +...
rho1f = coherentStateGenerate(alpha=photon_state['alpha1'], n=photon_state['n_states'])
rho2f = coherentStateGenerate(alpha=photon_state['alpha2'], n=photon_state['n_states'])

# Create density matrix
rho_0f = np.array([np.outer(rho1f, rho1f), np.outer(rho2f, rho2f)])


# Initial cavity1-cavity2 photon statistic matrix
prob_0 = np.outer(np.diagonal(rho_0f[0]), np.diagonal(rho_0f[1]))
