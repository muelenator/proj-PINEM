### !/usr/bin/env python3
# $HOME/envs/py37 python3

import numpy as np
from datetime import datetime
import argparse

# import logging
# logging.basicConfig(filename='/storage/home/rjm6826/SubmitTest1.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.info("Before mpi call hit.")

import os
import glob
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from bkgndLibs.PINEM import *
from bkgndLibs.electronFns import *
from bkgndLibs.T_modes_cylinder import *
from PINEM_tests.config.savepaths import default_dir, savedir

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank==0:
    print("Started script with {} processes".format(size), flush=True)

list_of_files = glob.glob(savedir)
# savefile = max(list_of_files, key=os.path.getctime)
# Keep only files with 'tcav' in the file name
list_of_files = [f for f in list_of_files if 'tcav' in f]
default_file = max(list_of_files, key=os.path.getmtime)
########################################################################################################
# Import data and config
# Create the parser
parser = argparse.ArgumentParser(description='Process some integers.')

# Add the arguments
parser.add_argument('--data_path', type=str, default=default_dir,
                    help='The path to the data directory')
parser.add_argument('--file_name', type=str, default=default_file, 
                    help='The name of the file to load')

parser.add_argument('--m_measurements', type=int, default=100, 
                    help='The number of measurements to take')
parser.add_argument('--n_times', type=int, default=100,
                    help='The number of times to run the simulation')

parser.add_argument('--E_range', type=str, default='[0,-1]', 
                    help='The indices of the range of electron energies to use')
parser.add_argument('--L_range', type=str, default='[0,-1]',
                    help='The indices of the range of free space lengths to use')
parser.add_argument('--t_range', type=str, default='[0,0]',
                    help='The indices of the range of time delays to use')
# Parse the arguments
args = parser.parse_args()
if args.file_name is None:
    raise ValueError("Please provide a file name to load")

# Use the argument
data_path = args.data_path
file_name = args.file_name
m_measurements = args.m_measurements
n_times = args.n_times
E_range = args.E_range
L_range = args.L_range
t_range = args.t_range

with h5py.File(file_name, 'r') as hf:
    prob_mats = np.zeros(hf['Probability'][:].shape, dtype='complex')
    prob_mats[:,:,:] = np.copy(hf['Probability'][:])

    rho_mats = np.copy(hf['Final State'])
    loaded_metadata = {key: hf['metadata'].attrs[key] for key in hf['metadata'].attrs}

########################################################################################################
Es = np.linspace(loaded_metadata['Maximum electron energy'] - loaded_metadata['Electron energy range'], 
                 loaded_metadata['Maximum electron energy'], 
                 loaded_metadata['Number of electron energies'])
Lfree = np.linspace(loaded_metadata['Phase minimum multiplier']*loaded_metadata['Phase reference'], 
                    loaded_metadata['Phase maximum multiplier']*loaded_metadata['Phase reference'], 
                    loaded_metadata['Number of free space lengths'])
# Time delay (between cavity1 and cavity 2 photons)
t_delay_ref = Lfree / EnT_to_vvec(loaded_metadata['Maximum electron energy'],0,True)[2]

t_delay = np.zeros((loaded_metadata['Number of free space lengths'], 
                     loaded_metadata['Number of time delays']))

for i, l in enumerate(Lfree):
    t_delay[i,:] = np.linspace(loaded_metadata['Minimum time delay multiplier']*t_delay_ref[i],
                            loaded_metadata['Maximum time delay multiplier']*t_delay_ref[i],
                            loaded_metadata['Number of time delays'])
########################################################################################################
# Parse the ranges based on parser args
Es_p = Es[int(E_range.split(',')[0].strip('[')):int(E_range.split(',')[1].strip(']'))]

if len(Lfree) > 1:
    # print(int(L_range.split(',')[0].strip('[')), int(L_range.split(',')[1].strip(']')), flush=True)
    l_lower = int(L_range.split(',')[0].strip('['))
    l_upper = int(L_range.split(',')[1].strip(']'))
    Lfree_p = Lfree[l_lower:l_upper]
else:
    Lfree_p = Lfree

if t_delay.shape[1] > 1:
    t_lower = int(t_range.split(',')[0].strip('['))
    t_upper = int(t_range.split(',')[1].strip(']'))
    t_delay_p = t_delay[:][t_lower:t_upper]
else:
    t_delay_p = t_delay

# print(Es_p.shape, Lfree_p.shape, t_delay_p.shape, flush=True)
########################################################################################################
# combinations0 = np.array(np.meshgrid(Es, Lfree)).T.reshape(-1,2)
# Reshape combinations0 from [[a,b,c], [d,e,f], ...] to [a,b,c,d,e,f,...]
combinations0 = [(e, l, t) for e in Es_p \
                 for l, t_row in zip(Lfree_p, t_delay_p) \
                    for t in t_row]
combinations0 = np.array(combinations0)

combinations_OG = [(e, l, t) for e in Es \
                        for l, t_row in zip(Lfree, t_delay) \
                            for t in t_row]
combinations_OG = np.array(combinations_OG)

if rank==0:
    combinations = combinations0.reshape(-1)
    meas = np.zeros((len(combinations0), 3, n_times))
    print(len(combinations0), flush=True)
    # print(combinations0, flush=True)
else:
    combinations = None
    meas = None

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

# Scatter the combinations
comm.Scatterv([combinations, sendcounts, displs, MPI.DOUBLE], data, root=0)
data = data.reshape(-1,3)

########################################################################################################
# Measure storage
meas_storage = np.zeros((len(data), 3, n_times))

for ct, (E, l, t) in enumerate(data):
    # i = np.argwhere((Es==E))
    # j = np.argwhere((Lfree==l))
    # print(np.argwhere(np.isclose(t_delay[j],t)), flush=True)
    # print(np.argwhere(np.isclose(t_delay[j][0],t)), flush=True)
    # print(np.argwhere(np.isclose(t_delay[j][0][0],t)), flush=True)
    t_ct = 0 #np.argwhere((t_delay[j][0]==t))[0][0]

    p_ind = np.where(np.all(np.isclose(combinations_OG, np.array([E, l, t]), atol=1e-4), 
                            axis=1))[0][0]
    # p_ind = np.where(np.all(combinations_OG == np.array([E, l, t]), axis=1))[0][0]
    rho = np.abs(prob_mats[p_ind])

    meas_storage[ct] = threeMeasSamp(rho, m_measurements, n_times)

    # If the bins_g2ct is all zeros, print that there is a problem
    if np.all(np.isclose(meas_storage[ct][0], 0)):
        print("Histogram g2 is all zeros for data_ct {} on rank {}".format(ct, rank), flush=True)


    print("Node {}, process {}".format(os.environ['SLURMD_NODENAME'], rank) \
           + " has COMPLETED {}/{}".format((ct+1), len(data)) \
            + ". Is done with E={:.2f}, l={:.2f}, ind={}.".format(E, l, p_ind), flush=True)

########################################################################################################
comm.Barrier()

if rank != 0:
    # print(meas_storage.shape, flush=True)
    comm.Send([meas_storage, MPI.DOUBLE], dest=0)
    # print("Node {}, process {} is sending data to root".format(os.environ['SLURMD_NODENAME'], rank), flush=True)

# comm.Barrier()

if rank == 0:
    print("Started data collection at the root node.", flush=True)

    for sp in range(1, size):
        meas_storage2 = np.zeros((len(to_send[sp]), 3, n_times))

        comm.Recv([meas_storage2, MPI.DOUBLE], source=sp)
        for ct, (E, l, t) in enumerate(to_send[sp]):
            p_ind = np.where(np.all(combinations0 == np.array([E, l, t]), axis=1))[0][0]
            # print(p_ind)
            meas[p_ind] = meas_storage2[ct,:,:]

    for ct, (E, l, t) in enumerate(to_send[0]):
        p_ind = np.where(np.all(combinations0 == np.array([E, l, t]), axis=1))[0][0]

        meas[p_ind] = meas_storage[ct,:,:]
########################################################################################################
# comm.Barrier()

if rank == 0:
    # Save data
    todays_date = datetime.today().strftime('%m/%d/%Y')

    metadata = dict(ndHist_date=todays_date)
    metadata.update(loaded_metadata)
    metadata.update(dict(E_range=E_range, L_range=L_range, t_range=t_range,
                    m_measurements=m_measurements, n_times=n_times))

    # Save the simulation date as a string in format %y%m%d
    sim_date = loaded_metadata['date']
    sim_date = sim_date.strip('/')
    sim_date = sim_date.strip(r'/')
    sim_date = sim_date.replace('/', '')
    sim_date = sim_date.replace(r'/','')

    savefile = data_path + sim_date + '_nDHist' + '.h5'

    h5store(savefile, meas, 'nDHist', metadata)


if rank==0:
    print("Job completed! Time is {}".format(datetime.now()), flush=True)

comm.Barrier()
MPI.Finalize()