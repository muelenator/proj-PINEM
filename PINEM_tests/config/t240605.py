# This config file...

import numpy as np
from bkgndLibs.physicalConstants import *
from bkgndLibs.PINEM import FS_phi
from bkgndLibs.electronFns import EnT_to_vvec

##########################################################################################
# Import photonic state
from PINEM_tests.initial_states.coherent_alpha_0_5 import photon_state, rho_0f, prob_0

# Import cavity fields
from PINEM_tests.waveguides.Kfir import wg, mode, Ez

# Import coupling profile
from PINEM_tests.couplingProfiles.photonsPlaneWaved import *
##########################################################################################
'''
Description of config variables:
    name: str
        Name of the configuration
    date: str
        Date of the configuration
    author: str
        Author of the configuration
    description: str
        Description of the configuration
    Number of electron energies: int
        Number of electron energies to simulate
    Electron energy range: int
        Range of electron energies to simulate (in eV)
    Enforce phase: bool
        Option to set the timing of the second cavity pulse to be in phase with the 
        maximally coupled electron.
        Couplings with other energies will then be sub-maximum for their respective energy.
    Constant coupling: bool
        Option to set the coupling strength as constant
    Reference coupling: float
        Reference coupling strength if constant coupling is True
'''

config = {
    'file_name': 't240605',
    'date': '24/06/05',
    'author': 'Rick Mueller',
    \
    'Save rho?': False,
    'Save coupling?': False,
    \
    'Number of electrons': 1,
    'Number of cavities': 2,
    \
    'Number of electron energies': 80,
    'Maximum electron energy': wg['e_E_max_g'] + 5e3,
    'Electron energy range': 10e3,
    \
    'Number of free space lengths': 120,
    'Phase reference': 1,
    # 'Phase reference': 1 / FS_phi(wg['omega'], EnT_to_vvec(wg['e_E_max_g'],0,True)[2], 1) / 2 * np.pi,
    'Phase minimum multiplier': .01,
    'Phase maximum multiplier': 1.,
    \
    'Number of time delays': 1,
    'Minimum time delay multiplier': .95,
    'Maximum time delay multiplier': 1.,
}
##########################################################################################
########## EXPLICIT VARIABLES ############################################################
# Electron energy range
Es = np.linspace(config['Maximum electron energy'] - config['Electron energy range'], 
                 config['Maximum electron energy'], 
                 config['Number of electron energies'])

# Free space propogation length range
Lfree = np.linspace(config['Phase minimum multiplier']*config['Phase reference'], 
                    config['Phase maximum multiplier']*config['Phase reference'], 
                    config['Number of free space lengths'])

# Time delay (between cavity1 and cavity 2 photons)
t_delay_ref = Lfree / EnT_to_vvec(config['Maximum electron energy'],0,True)[2]
# t_delay_ref = 0*Lfree

t_delay = np.zeros((config['Number of free space lengths'], 
                     config['Number of time delays']))

for i, l in enumerate(Lfree):
    t_delay[i,:] = np.linspace(config['Minimum time delay multiplier']*t_delay_ref[i],
                            config['Maximum time delay multiplier']*t_delay_ref[i],
                            config['Number of time delays'])

##########################################################################################
########## COUPLING CALCULATION ##########################################################
# Define coupling strength array (one for each cavity)
gQ = calc_gQ(Ez, mode, wg, config, Es, Lfree, t_delay)
# print(gQ)
# gQ = np.zeros((config['Number of electron energies'], 
#                config['Number of free space lengths'],
#                config['Number of time delays'],
#                config['Number of cavities']))
# Currently has format (Number of electron energies, Number of free space lengths,
#                        Number of time delays, Number of cavities)
             
        
##########################################################################################