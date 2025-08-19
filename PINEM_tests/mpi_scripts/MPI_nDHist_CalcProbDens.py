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
list_of_files = [f for f in list_of_files if 'nDHist' in f]
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

# parser.add_argument('--m_measurements', type=int, default=100, 
#                     help='The number of measurements to take')
# parser.add_argument('--n_times', type=int, default=100,
#                     help='The number of times to run the simulation')

# parser.add_argument('--E_range', type=str, default='[0,-1]', 
#                     help='The indices of the range of electron energies to use')
# parser.add_argument('--L_range', type=str, default='[0,-1]',
#                     help='The indices of the range of free space lengths to use')
# parser.add_argument('--t_range', type=str, default='[0,0]',
#                     help='The indices of the range of time delays to use')
# Parse the arguments
args = parser.parse_args()
if args.file_name is None:
    raise ValueError("Please provide a file name to load")

# Use the argument
data_path = args.data_path
file_name = args.file_name
# m_measurements = args.m_measurements
# n_times = args.n_times
# E_range = args.E_range
# L_range = args.L_range
# t_range = args.t_range

with h5py.File(file_name, 'r') as hf:
    meas_in = np.zeros(hf['nDHist'][:].shape)
    # if rank==0: print(meas_in.shape, flush=True)
    meas_in[:,:,:] = np.copy(hf['nDHist'][:])
    loaded_metadata = {key: hf['metadata'].attrs[key] for key in hf['metadata'].attrs}

########################################################################################################
# Es = np.linspace(loaded_metadata['Maximum electron energy'] - loaded_metadata['Electron energy range'], 
#                  loaded_metadata['Maximum electron energy'], 
#                  loaded_metadata['Number of electron energies'])
# Lfree = np.linspace(loaded_metadata['Phase minimum multiplier']*loaded_metadata['Phase reference'], 
#                     loaded_metadata['Phase maximum multiplier']*loaded_metadata['Phase reference'], 
#                     loaded_metadata['Number of free space lengths'])
# # Time delay (between cavity1 and cavity 2 photons)
# t_delay_ref = Lfree / EnT_to_vvec(loaded_metadata['Maximum electron energy'],0,True)[2]

# t_delay = np.zeros((loaded_metadata['Number of free space lengths'], 
#                      loaded_metadata['Number of time delays']))


# for i, l in enumerate(Lfree):
#     t_delay[i,:] = np.linspace(loaded_metadata['Minimum time delay multiplier']*t_delay_ref[i],
#                             loaded_metadata['Maximum time delay multiplier']*t_delay_ref[i],
#                             loaded_metadata['Number of time delays'])
########################################################################################################
# Parse the ranges based on parser args
# Es_p = Es[int(E_range.split(',')[0].strip('[')):int(E_range.split(',')[1].strip(']'))]

# if len(Lfree) > 1:
#     # print(int(L_range.split(',')[0].strip('[')), int(L_range.split(',')[1].strip(']')), flush=True)
#     Lfree_p = Lfree[int(L_range.split(',')[0].strip('[')):int(L_range.split(',')[1].strip(']'))]
# else:
#     Lfree_p = Lfree

# if t_delay.shape[1] > 1:
#     t_delay_p = t_delay[:][int(t_range.split(',')[0].strip('[')):int(t_range.split(',')[1].strip(']'))]
# else:
#     t_delay_p = t_delay

# print(Es_p.shape, Lfree_p.shape, t_delay_p.shape, flush=True)
########################################################################################################
# combinations0 = np.array(np.meshgrid(Es, Lfree)).T.reshape(-1,2)
# Reshape combinations0 from [[a,b,c], [d,e,f], ...] to [a,b,c,d,e,f,...]
# combinations0 = [(e, l, t) for e in Es_p \
#                  for l, t_row in zip(Lfree_p, t_delay_p) \
#                     for t in t_row]
# combinations0 = np.array(combinations0)
combinations = np.arange(meas_in.shape[0], dtype='float64')

if rank==0:
    # combinations = combinations0.reshape(-1)
    # meas = np.zeros((len(combinations), 3, n_times))
    # print(combinations0, flush=True)
    # combinations = range(meas_in.shape[0])
    meas = np.zeros((len(combinations), 3, meas_in.shape[1]))
else:
    # combinations = None
    meas = None

# Split the combinations into 'size' parts
to_send = np.array_split(combinations, size)

nto_send = [np.zeros(len(x.reshape(-1))) for x in to_send]

data = nto_send[rank]

# Use scatterv to distribute the combinations to the processes
sendcounts = [len(x.reshape(-1)) for x in to_send]
displs = [int(np.sum(sendcounts[:i])) for i in range(len(sendcounts))]

# Convert sendcounts and displacements to tuples
sendcounts = tuple(sendcounts)
displs = tuple(displs)

# print(sendcounts, displs, flush=True)

# Scatter the combinations
comm.Scatterv([combinations, sendcounts, displs, MPI.DOUBLE], data, root=0)
# data = data.reshape(-1,3)
data = np.array(data).astype(int)

########################################################################################################
# Storage of bins (0-2), number of bins (3-5)
mean_storage = np.zeros((3, len(data)))
nbins_storage = np.zeros((3, len(data)))
bin_bds = np.zeros((3, len(data), 2))
bin_bws = np.zeros((3, len(data)))

# bins_g2 = np.array([])
# bins_n2 = np.array([])
# bins_qmi = np.array([])

for this_ct, data_ct in enumerate(data):
    # i = np.argwhere((Es==E))
    # j = np.argwhere((Lfree==l))
    # print(np.argwhere(np.isclose(t_delay[j],t)), flush=True)
    # print(np.argwhere(np.isclose(t_delay[j][0],t)), flush=True)
    # print(np.argwhere(np.isclose(t_delay[j][0][0],t)), flush=True)
    # t_ct = 0 #np.argwhere((t_delay[j][0]==t))[0][0]
    mean_storage[0][this_ct] = np.mean(meas_in[data_ct][0])
    mean_storage[1][this_ct] = np.mean(meas_in[data_ct][1])
    mean_storage[2][this_ct] = np.mean(meas_in[data_ct][2])

    _, bins_g2ct = np.histogram(meas_in[data_ct][0], bins='fd', density=True)
    _, bins_n2ct = np.histogram(meas_in[data_ct][1], bins='fd', density=True)
    _, bins_qmict = np.histogram(meas_in[data_ct][2], bins='fd', density=True)
    
    # bins_g2 = np.concatenate([bins_g2, bins_g2ct], axis=None)
    # bins_n2 = np.concatenate([bins_n2, bins_n2ct], axis=None)
    # bins_qmi = np.concatenate([bins_qmi, bins_qmict], axis=None)
    
    nbins_storage[0][this_ct] = bins_g2ct.shape[0]
    nbins_storage[1][this_ct] = bins_n2ct.shape[0]
    nbins_storage[2][this_ct] = bins_qmict.shape[0]

    bin_bds[0][this_ct] = [bins_g2ct[0], bins_g2ct[-1]]
    bin_bds[1][this_ct] = [bins_n2ct[0], bins_n2ct[-1]]
    bin_bds[2][this_ct] = [bins_qmict[0], bins_qmict[-1]]

    bin_bws[0][this_ct] = (bins_g2ct[1] - bins_g2ct[0])
    bin_bws[1][this_ct] = (bins_n2ct[1] - bins_n2ct[0])
    bin_bws[2][this_ct] = (bins_qmict[1] - bins_qmict[0])

    # If the bins_g2ct is all zeros, print that there is a problem
    if np.all(np.isclose(meas_in[data_ct][0],0)):
        print("Histogram g2 is all zeros for data_ct {} on rank {}".format(data_ct, rank), flush=True)

    print("Node {}, process {}".format(os.environ['SLURMD_NODENAME'], rank) \
            + " is done with job {} of {}".format(data_ct, meas_in.shape[0]), flush=True)

########################################################################################################
# Send the arrays of fixed length first
if rank != 0:
    comm.Send([mean_storage, MPI.DOUBLE], dest=0)
    comm.Send([nbins_storage, MPI.DOUBLE], dest=0)
    comm.Send([bin_bds, MPI.DOUBLE], dest=0)
    comm.Send([bin_bws, MPI.DOUBLE], dest=0)

# comm.Barrier()

if rank == 0:
    mean_storage2 = np.zeros((3, meas_in.shape[0]))
    nbins_storage2 = np.zeros((3, meas_in.shape[0]))
    bin_bds_storage2 = np.zeros((3, meas_in.shape[0], 2))
    bin_bws_storage2 = np.zeros((3, meas_in.shape[0]))

    for sp in range(1, size):
        mean_storage_int = np.zeros((3, len(to_send[sp])))
        nbins_storage_int = np.zeros((3, len(to_send[sp])))
        bin_bds_int = np.zeros((3, len(to_send[sp]), 2))
        bin_bws_int = np.zeros((3, len(to_send[sp])))

        comm.Recv([mean_storage_int, MPI.DOUBLE], source=sp)
        comm.Recv([nbins_storage_int, MPI.DOUBLE], source=sp)
        comm.Recv([bin_bds_int, MPI.DOUBLE], source=sp)
        comm.Recv([bin_bws_int, MPI.DOUBLE], source=sp)

        for this_ct, data_ct in enumerate(to_send[sp]):
            mean_storage2[:,int(data_ct)] = mean_storage_int[:,this_ct]
            nbins_storage2[:,int(data_ct)] = nbins_storage_int[:,this_ct]
            bin_bds_storage2[:,int(data_ct)] = bin_bds_int[:,this_ct,:]
            bin_bws_storage2[:,int(data_ct)] = bin_bws_int[:,this_ct]

    for this_ct, data_ct in enumerate(to_send[0]):
        mean_storage2[:,int(data_ct)] = mean_storage[:,this_ct]
        nbins_storage2[:,int(data_ct)] = nbins_storage[:,this_ct]
        bin_bds_storage2[:,int(data_ct)] = bin_bds[:,this_ct,:]
        bin_bws_storage2[:,int(data_ct)] = bin_bws[:,this_ct]
########################################################################################################
# comm.Barrier()
# # Gather the lengths of arrays from all processes
# lengths1 = comm.gather(len(bins_g2), root=0)
# lengths2 = comm.gather(len(bins_n2), root=0)
# lengths3 = comm.gather(len(bins_qmi), root=0)

# if rank==0: 
#     print(sum(lengths1), flush=True)
#     print(sum(nbins_storage2[0]), flush=True)

# if rank == 0:
#     # Calculate the total length of each gathered array
#     total_length1 = sum(lengths1)
#     total_length2 = sum(lengths2)
#     total_length3 = sum(lengths3)

#     # Prepare the arrays to receive data from all processes
#     gathered_array1 = np.zeros(total_length1, dtype=float)
#     gathered_array2 = np.zeros(total_length2, dtype=float)
#     gathered_array3 = np.zeros(total_length3, dtype=float)

#     # Create displacement arrays for Gatherv
#     displacements1 = [sum(lengths1[:i]) for i in range(size)]
#     displacements2 = [sum(lengths2[:i]) for i in range(size)]
#     displacements3 = [sum(lengths3[:i]) for i in range(size)]
# else:
#     gathered_array1 = None
#     gathered_array2 = None
#     gathered_array3 = None
#     displacements1 = None
#     displacements2 = None
#     displacements3 = None

# # Use Gatherv to gather arrays with variable lengths
# comm.Gatherv(sendbuf=bins_g2, recvbuf=(gathered_array1, lengths1, displacements1, MPI.DOUBLE), root=0)
# comm.Gatherv(sendbuf=bins_n2, recvbuf=(gathered_array2, lengths2, displacements2, MPI.DOUBLE), root=0)
# comm.Gatherv(sendbuf=bins_qmi, recvbuf=(gathered_array3, lengths3, displacements3, MPI.DOUBLE), root=0)

# if rank==0:
#     bins_g2p = np.array([])
#     bins_n2p = np.array([])
#     bins_qmip = np.array([])

#     # Re-sort the arrays based on the scattered combination data_ct
#     for i in combinations:
#         for sp in range(0, size):
#             if i in to_send[sp]:
#                 ind = np.where(to_send[sp]==i)[0][0]
#                 print(ind, flush=True)

#                 lowerg2 = int(sum(nbins_storage2[0][:ind-1])) if ind!=0 else 0
#                 upperg2 = int(sum(nbins_storage2[0][:ind]))

#                 g2_add = gathered_array1[displacements1[sp]:displacements1[sp]+lengths1[sp]]
#                 g2_add = g2_add[lowerg2:upperg2]

#                 lowern2 = int(sum(nbins_storage2[1][:ind-1])) if ind!=0 else 0
#                 uppern2 = int(sum(nbins_storage2[1][:ind]))

#                 n2_add = gathered_array2[displacements2[sp]:displacements2[sp]+lengths2[sp]]
#                 n2_add = n2_add[lowern2:uppern2]

#                 lowerqmi = int(sum(nbins_storage2[2][:ind-1])) if ind!=0 else 0
#                 upperqmi = int(sum(nbins_storage2[2][:ind]))

#                 qmi_add = gathered_array3[displacements3[sp]:displacements3[sp]+lengths3[sp]]
#                 qmi_add = qmi_add[lowerqmi:upperqmi]

#                 bins_g2p = np.concatenate([bins_g2p, g2_add], axis=None)
#                 bins_n2p = np.concatenate([bins_n2p, n2_add], axis=None)
#                 bins_qmip = np.concatenate([bins_qmip, qmi_add], axis=None)
#                 break

########################################################################################################
# comm.Barrier()

if rank == 0:
    # print(bins_g2p.shape, flush=True)

    # Save data
    todays_date = datetime.today().strftime('%m/%d/%Y')

    metadata = dict(ndHist_date=todays_date)
    metadata.update(loaded_metadata)
    # metadata.update(dict(E_range=E_range, L_range=L_range, t_range=t_range,
    #                 m_measurements=m_measurements, n_times=n_times))

    # Save the simulation date as a string in format %y%m%d
    sim_date = loaded_metadata['date']
    sim_date = sim_date.strip('/')
    sim_date = sim_date.strip(r'/')
    sim_date = sim_date.replace('/', '')
    sim_date = sim_date.replace(r'/','')

    savefile = data_path + sim_date + '_nDProbDensHist' + '.h5'

    h5store(savefile, mean_storage2, 'means', metadata)
    h5store(savefile, nbins_storage2, 'nbins', metadata, mode='a', md_store=False)
    # h5store(savefile, bins_g2p, 'bins_g2', metadata, mode='a', md_store=False)
    # h5store(savefile, bins_n2p, 'bins_n2', metadata, mode='a', md_store=False)
    # h5store(savefile, bins_qmip, 'bins_qmi', metadata, mode='a', md_store=False)
    h5store(savefile, bin_bds_storage2, 'bin_bds', metadata, mode='a', md_store=False)
    h5store(savefile, bin_bws_storage2, 'bin_bws', metadata, mode='a', md_store=False)

if rank==0:
    print("Job completed! Time is {}".format(datetime.now()), flush=True)

comm.Barrier()
MPI.Finalize()