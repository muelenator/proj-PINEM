### !/usr/bin/env python3
# $HOME/envs/py37 python3

import numpy as np
from datetime import datetime

# import logging
# logging.basicConfig(filename='/storage/home/rjm6826/SubmitTest1.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.info("Before mpi call hit.")

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from bkgndLibs.PINEM import *
from bkgndLibs.electronFns import *
from bkgndLibs.T_modes_cylinder import *
from PINEM_tests.config.savepaths import data_path

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank==0:
    print("Started script with {} processes".format(size), flush=True)

########################################################################################################
# Import config
# from PINEM_tests.config.t240515_1 import *
from PINEM_tests.config.t241031 import *
########################################################################################################
n_states = photon_state['n_states']
########################################################################################################
maxFockNum = n_states
mk = maxFockNum
k_range = range(-mk,mk+1)

########################################################################################################
# combinations0 = np.array(np.meshgrid(Es, Lfree)).T.reshape(-1,2)
# Reshape combinations0 from [[a,b,c], [d,e,f], ...] to [a,b,c,d,e,f,...]
combinations0 = [(e, l, t) for e in Es \
                 for l, t_row in zip(Lfree, t_delay) \
                    for t in t_row]
combinations0 = np.array(combinations0)

if rank==0:
    combinations = combinations0.reshape(-1)
    prob_mats = np.zeros((len(combinations0),)+ prob_0.shape, dtype='complex')
    rho_mats = np.zeros((len(combinations0), ) + (n_states**2, n_states**2), dtype='complex')
else:
    combinations = None
    prob_mats = None
    rho_mats = None

# Split the combinations into 'size' parts
to_send = np.array_split(combinations0, size)

nto_send = [np.zeros(len(x.reshape(-1))) for x in to_send]

data = nto_send[rank]

# Use scatterv to distribute the combinations to the processes
sendcounts = [len(x.reshape(-1)) for x in to_send]
displs = [int(np.sum(sendcounts[:i])) for i in range(len(sendcounts))]

# Convert sendcounts and displacements to tuples
sendcounts = tuple(sendcounts)
displs = tuple(displs)
# print(sendcounts)
# print(displs)
# print(combinations)
# print(data)

# Scatter the combinations
comm.Scatterv([combinations, sendcounts, displs, MPI.DOUBLE], data, root=0)
# comm.Scatterv(combinations, data, root=0)
# comm.Scatter(combinations, data, root=0)
data = data.reshape(-1,3)

########################################################################################################
# print("Process {} has data {}".format(rank, data))

# Probability matrix storage
pm_storage = np.zeros(prob_0.shape + (len(data), ), dtype='complex')

# Final state matrix storage
fs_storage = np.zeros((n_states**2, n_states**2, len(data)), dtype='complex')

for ct, (E, l, t) in enumerate(data):
    i = np.argwhere((Es==E))
    j = np.argwhere((Lfree==l))
    t_ct = np.argwhere((t_delay[j][0][0]==t))[0][0]

    temp_prob_mat = np.zeros_like(prob_0, dtype='complex')

    # Find the Kraus operators
    k_mats = np.zeros((2, n_states, n_states, ) +(2*maxFockNum-1, len(k_range),), dtype='complex') #,dtype = 'object')
    
    # Find the phase accumulated by the electron
    v_E = EnT_to_vvec(E,0,True)[2]
    phie = FS_phi(wg['omega'], v_E, l) # Free space propagation constant of electron
    phie = float(phie)

    # print(gQarr[g][i,j,1][0][0], float(gQarr[g][i,j,1][0][0]), phie, float(phie), float(phie[0]))
    # print(E,g,l)
    for k_ct,k in enumerate(k_range):
        k_mats[:,:,:,:,k_ct]= A_k(maxFockNum, gQ[i, j, t_ct, 0][0][0], gQ[i, j, t_ct, 1][0][0], 
                                (2, n_states, n_states), k, phi=phie)

    # Put the Kraus operators in tensor product form
    A_kT = krause_toTProd(rho_0f, maxFockNum, k_range, k_mats, rhoNotTProd=photon_state['Not a tensor product'])

    # Apply the Kraus operators to the initial state
    temp_rho_mat, temp_prob_mat = krause_apply(rho_0f.astype('complex'), k_range, A_kT, 
                                                config['Number of electrons'], 
                                                rhoNotTProd=photon_state['Not a tensor product'])

    # comm.Gatherv(temp_prob_mat, [prob_mats[i,j,g],, root=0)
    pm_storage[:,:,ct] = np.copy(temp_prob_mat)
    fs_storage[:,:,ct] = np.copy(temp_rho_mat)

    # sys.stdout.write("Done with E={:.2f}, g={:.2f}, l={:.2f}".format(E,gQarr[g][i,j,1][0][0], l))
    print("Node {}, process {}".format(os.environ['SLURMD_NODENAME'], rank) \
            + " is done with E={:.2f}, g={:.2f}, l={:.2f}".format(E, gQ[i,j,t_ct,1][0][0], l), flush=True)

    del k_mats, A_kT
########################################################################################################
if rank != 0:
    comm.Send(pm_storage, dest=0)

else:
    for sp in range(1, size):
        pm_storage2 = np.zeros(prob_0.shape + (len(to_send[sp]), ), dtype='complex')
        fs_storage2 = np.zeros((n_states**2, n_states**2, len(to_send[sp])), dtype='complex')

        comm.Recv(pm_storage2, source=sp)
        for ct, (E, l, t) in enumerate(to_send[sp]):
            # i = np.argwhere((Es==E))
            # j = np.argwhere((Lfree==l))
            # t_ct = np.argwhere((t_delay[j]==t))

            # p_ind = np.argwhere((combinations0 == [E, l, t]))
            p_ind = np.where(np.all(combinations0 == np.array([E, l, t]), axis=1))[0][0]
            # print(p_ind, flush=True)

            prob_mats[p_ind] = pm_storage2[:,:,ct]
            rho_mats[p_ind] = fs_storage2[:,:,ct]

    for ct, (E, l, t) in enumerate(to_send[0]):
        # p_ind = np.argwhere((combinations0 == [E, l, t]))
        p_ind = np.where(np.all(combinations0 == np.array([E, l, t]), axis=1))[0][0]

        prob_mats[p_ind] = pm_storage[:,:,ct]
        rho_mats[p_ind] = fs_storage[:,:,ct]
########################################################################################################
comm.Barrier()

if rank == 0:
    # Save data
    todays_date = datetime.today().strftime('%m/%d/%Y')

    metadata = dict(date=todays_date)
    metadata.update(config)
    metadata.update(photon_state)
    metadata.update(wg)
    metadata.update(env_fn)

    todays_date_time = datetime.today().strftime('%m%d%y_%H%M%S')

    savefile = data_path + todays_date_time + '_tcav' + '.h5'

    h5store(savefile, prob_mats.astype('complex'), 'Probability', metadata)

    if config['Save rho?']:
        h5store(savefile, rho_mats.astype('complex'), 'Final State', metadata, mode='a', md_store=False)

    if config['Save coupling?']:
        h5store(savefile, gQ, 'Coupling', metadata, mode='a', md_store=False)

comm.Barrier()

if rank==0:
    print("Job completed! Time is {}".format(datetime.now()), flush=True)

MPI.Finalize()