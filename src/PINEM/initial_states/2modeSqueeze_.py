# This is a two-part state of light. 
# The states are entangled
# The two 'parts' (functionally, cavities) are loaded with a two-mode squeezed state
    # Here, alpha = .5

from bkgndLibs.physicalConstants import *
from bkgndLibs.PINEM import twomodesqueezedStateGenerate
import numpy as np

# Number of Fock states (this is the same for both cavities)
n_states = 15
alpha = .5
phi = 0

# State info
photon_state = {
    'type': 'two-mode-squeezed',
    'alpha': alpha,
    'phi': phi,
    'n_states': n_states,
}

# Fock space, rho = a|0> + b|1> +...
# Create density matrix
rho_0f = twomodesqueezedStateGenerate(alpha, phi, n_states)


# Initial cavity1-cavity2 photon statistic matrix
prob_0 = np.outer(np.diagonal(rho_0f[0]), np.diagonal(rho_0f[1]))

p_shape = prob_0.shape