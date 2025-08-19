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
    'file_name': 't240515_1',
    'date': '24/05/2015',
    'author': 'rjm','Number of electron energies': 20,
    'Number of electrons': 1,
    'Number of electron energies': 10,
    'Number of free space lengths': 10,
    'Maximum electron energy': wg['e_E_max_g'],
    'Electron energy range': 30,
    'Enforce phase': True,
    'Constant coupling': False,
    'Reference coupling': 0,
    'Phase reference': 1 / FS_phi(wg['omega'], EnT_to_vvec(wg['e_E_max_g'],0,True)[2], 1) / 2 * np.pi,
    'Phase minimum multiplier': .1,
    'Phase maximum multiplier': 2
}
##########################################################################################
# Electron energy range
Es = np.linspace(config['Maximum electron energy'] - config['Electron energy range'], 
                 config['Maximum electron energy'], 
                 config['Number of electron energies'])
vs = EnT_to_vvec(Es,0,True)[2] # Relativstic velocity of electrons (m/s)
taus = wg['Lcav'] / vs # Time it takes the electron to traverse the cavity

# Free space propogation length range
Lfree = np.linspace(config['Phase minimum multiplier']*config['Phase reference'], 
                    config['Phase maximum multiplier']*config['Phase reference'], 
                    config['Number of free space lengths'])

# Field amplitude
Ez_amp0 = Ez(0)

# Define coupling strength matrices (one for each cavity)
gQ = np.zeros((config['Number of electron energies'], 
               config['Number of free space lengths'],
               2))

# THIS VAR DESERVES TO BE INVESTIGATED
t_delay = 0 * Lfree

if not config['Constant coupling']:
    kz = mode.mode_params[(wg['m'],wg['l'],wg['type'],'kz')]

    Ez_ints1 = (1/(wg['omega'] - kz*vs)) \
                * np.sin((wg['omega'] - kz*vs)*taus)
    Ez_ints2 = np.zeros((len(taus), len(Lfree)))
    
    if config['Enforce phase']:
        rel_phase = np.zeros_like(Ez_ints2)

        for i, l in enumerate(Lfree):
            rel_phase[:,i] = -wg['omega'] * (l/vs - t_delay[i])
            Ez_ints2[:,i] = (1/(wg['omega'] - kz*vs)) \
                            * (np.sin((wg['omega'] - kz*vs)*taus
                                    + rel_phase[:,i] )
                                - np.sin(rel_phase[:,i]))
            
            # First cavity
            gQ[:,i,0] = e_c * vs * Ez_amp0 * Ez_ints1 / 2 / hbar_Js / wg['omega']
            # Second cavity
            gQ[:,i,1] = e_c * vs * Ez_amp0 * Ez_ints2[:,i] / 2 / hbar_Js / wg['omega']
    else:
        for i, l in enumerate(Lfree):
            # Both cavities
            gQ[:,i,:] = e_c * vs * Ez_amp0 * Ez_ints1 / 2 / hbar_Js / wg['omega']
            
else:
    gQ[:,:,:] = config['Reference coupling']
        
##########################################################################################