import numpy as np
from bkgndLibs.physicalConstants import *
from bkgndLibs.electronFns import EnT_to_vvec


'''
Coupling profile across all cavities:

'''
env_fn = {
    'name': 'Plane wave',
    'Enforce phase': False
}

# Integrate field along the electron trajectory
def calc_gQ(Ez, mode, wg, config, Es, Lfree, t_delay):
    gQ = np.zeros((config['Number of electron energies'], 
               config['Number of free space lengths'],
               config['Number of time delays'],
               config['Number of cavities']))
    # print(gQ.shape)

    vs = EnT_to_vvec(Es,0,True)[2] # Relativstic velocity of electrons (m/s)
    taus = wg['Lcav'] / vs # Time it takes the electron to traverse the cavity
    Ez_amp0 = Ez(0) # Field amplitude

    kz = mode.mode_params[(wg['m'],wg['l'],wg['type'],'kz')]

    Ez_ints1 = (1/(wg['omega'] - kz*vs)) \
                * np.sin((wg['omega'] - kz*vs)*taus)
    Ez_ints2 = np.zeros((config['Number of electron energies'], 
                        config['Number of free space lengths'],
                        config['Number of time delays']))

    if env_fn['Enforce phase']:
        rel_phase = np.zeros_like(Ez_ints2)

        for i, l in enumerate(Lfree):
            for j, t in enumerate(t_delay[i]):
                rel_phase[:,i,j] = -wg['omega'] * (l/vs - t)

                Ez_ints2[:,i,j] = (1/(wg['omega'] - kz*vs)) \
                                * ( np.sin((wg['omega'] - kz*vs)*taus + rel_phase[:,i,j] )
                                    - np.sin(rel_phase[:,i,j]))
                
                # First cavity
                gQ[:,i,j,0] = e_c * vs * Ez_amp0 * Ez_ints1 / 2 / hbar_Js / wg['omega']
                # Second cavity
                gQ[:,i,j,1] = e_c * vs * Ez_amp0 * Ez_ints2[:,i,j] / 2 / hbar_Js / wg['omega']
    else:
        for i, l in enumerate(Lfree):
            for j, t in enumerate(t_delay[i]):
                # Both cavities
                gQ[:,i,j,0] = e_c * vs * Ez_amp0 * Ez_ints1 / 2 / hbar_Js / wg['omega']
                gQ[:,i,j,1] = e_c * vs * Ez_amp0 * Ez_ints1 / 2 / hbar_Js / wg['omega']

    return gQ