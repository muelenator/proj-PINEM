# This is the configuration for the Ni02 fiber waveguide designed in 
# https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.103602
# for 200 keV electrons.

from bkgndLibs.physicalConstants import *
from bkgndLibs.T_modes_cylinder import *

wg = {
    'name': 'Kfir',
    'radius': 463e-9 / 2,
    'eps1': (2.0112)**2 * ep0_F_m,  # Ni02 core permittivity (cite?) (see also: 7.4 * ep0_F_m)
    'eps2': ep0_F_m,        # Free space permittivity
    'Lcav': 100e-6,         # Cavity length
    'type': 'HE',           # Mode type
    'm': 1,                 # Transverse mode number
    'l': 1,                 # Longitudinal mode number
    'omega': 1.166 / hbar_eVs, # or 2*np.pi * c_m_s / 1024e-9
    'e_E_max_g': 199117.50942468643 # Energy of a maximally coupled electron, eV
}

mode = T_modes_stepClad(wg['radius'],100,
                        wg['eps1'],wg['eps2'],
                        mu0, wg['Lcav'], cladModelled=False)

# Field in z experienced by the electron:
Ez = lambda z: mode.HE(wg['type'], 'E', wg['omega'], 1.01*wg['omega']*np.sqrt(mode.mu*mode.eps2), 
                       r=1.000001*mode.a1, phi=0, z=z, 
                       mm=wg['m'], ll=wg['l'], forNorm=False)[2].real