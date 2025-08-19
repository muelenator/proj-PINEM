### !/usr/bin/env python3
# $HOME/envs/py37 python3

import numpy as np
from datetime import datetime
import argparse
import matplotlib.pyplot as plt

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
from bkgndLibs.helpers import powerset
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
list_of_files1 = [f for f in list_of_files if 'nDProbDensHist' in f]
default_file1 = max(list_of_files1, key=os.path.getmtime)

list_of_files2 = [f for f in list_of_files if 'nDHist' in f]
default_file2 = max(list_of_files2, key=os.path.getmtime)
########################################################################################################
# Import data and config
# Create the parser
parser = argparse.ArgumentParser(description='')

# Add the arguments
parser.add_argument('--data_path', type=str, default=default_dir,
                    help='The path to the data directory')
parser.add_argument('--file_name1', type=str, default=default_file1, 
                    help='The name of the file to load')
parser.add_argument('--file_name2', type=str, default=default_file2, 
                    help='The name of the file to load')
# Add measures as a list of strings
parser.add_argument('--measures', nargs='+', type=str, default=['g2', 'n2', 'qmi'], 
                    help='The list of measurements to use. Options are g2, n2, qmi. \
                            Default is "[g2, n2, qmi]".')
measure_dict = {'g2': 0, 'n2': 1, 'qmi': 2}

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
# if args.file_name1 is None:
#     raise ValueError("Please provide a file name to load")

# Use the argument
data_path = args.data_path
file_name1 = args.file_name1
file_name2 = args.file_name2
# measures_req = args.measures.strip('[').strip(']').replace(" ", "").split(',')
measures_req = args.measures
# m_measurements = args.m_measurements
# n_times = args.n_times
# E_range = args.E_range
# L_range = args.L_range
# t_range = args.t_range

with h5py.File(file_name1, 'r') as hf:
    means = np.copy(hf['means'])
    nbins = np.copy(hf['nbins'])
    # bins_g2 = np.copy(hf['bins_g2']) #[::-1]
    # bins_n2 = np.copy(hf['bins_n2']) #[::-1]
    # bins_qmi = np.copy(hf['bins_qmi']) #[::-1]
    bin_bds = np.copy(hf['bin_bds'])
    bin_bws = np.copy(hf['bin_bws'])

    loaded_metadata = {key: hf['metadata'].attrs[key] for key in hf['metadata'].attrs}

with h5py.File(file_name2, 'r') as hf:
    # meas_in = np.zeros(hf['nDHist'][:].shape)
    # meas_in[:,:,:] = np.copy(hf['nDHist'][:])
    meas_in = np.copy(hf['nDHist'])
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

# if rank==0:
#     # combinations = combinations0.reshape(-1)
#     # meas = np.zeros((len(combinations), 3, n_times))
#     # print(combinations0, flush=True)
#     # combinations = range(meas_in.shape[0])
#     meas = np.zeros((len(combinations), 3, meas_in.shape[1]))

# else:
#     # combinations = None
#     meas = None


# Create the new bins
# Bin widths for each energy measurement
# print(bins_g2.shape)
# bw_g2 = []
# bw_n2 = []
# bw_qmi= []
# for i in range(meas_in.shape[0]):
#     # if i==0:
#     #     bw_g2.append(bins_g2[int(nbins[0][i])])
#     #     bw_n2.append(bins_n2[int(nbins[1][i])])
#     #     bw_qmi.append(bins_qmi[int(nbins[2][i])])

#     # else:
#     ind = int(i)

#     # lowerg2, upperg2 = int(sum(nbins[0][:i])) - 2, int(sum(nbins[0][:i]) - 1)
#     # lowern2, uppern2 = int(sum(nbins[1][:i])) - 2, int(sum(nbins[1][:i]) - 1)
#     # lowerqmi, upperqmi = int(sum(nbins[2][:i])) - 2, int(sum(nbins[2][:i]) - 1)
#     print("NBins", nbins[0][:ind], flush=True)

#     lowerg2 = int(sum(nbins[0][:ind])) - 2 if ind > 0 else 0
#     upperg2 = int(sum(nbins[0][:ind])) - 1 if ind > 0 else 1

#     lowern2 = int(sum(nbins[1][:ind])) - 2 if ind > 0 else 0
#     uppern2 = int(sum(nbins[1][:ind])) - 1 if ind > 0 else 1

#     lowerqmi = int(sum(nbins[2][:ind])) - 2 if ind > 0 else 0
#     upperqmi = int(sum(nbins[2][:ind])) - 1 if ind > 0 else 1

#     print(lowerg2, flush=True)


#     bwg2_val = abs(bins_g2[upperg2] - bins_g2[lowerg2])
#     bwn2_val = abs(bins_n2[uppern2] - bins_n2[lowern2])
#     bwqmi_val = abs(bins_qmi[upperqmi] - bins_qmi[lowerqmi])

#     bw_g2.append(bwg2_val)
#     bw_n2.append(bwn2_val)
#     bw_qmi.append(bwqmi_val)

# Compute the average bin widths
avg_bw_g2 = np.mean(bin_bws[0])
avg_bw_n2 = np.mean(bin_bws[1])
avg_bw_qmi = np.mean(bin_bws[2])

# Save a histogram of the g2 bin widths
plt.plot(bin_bws[0])
plt.savefig(data_path + 'bw_g2.png')

plt.clf()
plt.plot(bin_bws[2])
plt.savefig(data_path + 'bw_qmi.png')

plt.clf()
plt.plot(bin_bds[0,:,0])
plt.plot(bin_bds[0,:,1])
plt.savefig(data_path + 'bin_bds_g2.png')

plt.clf()
plt.plot(bin_bds[2,:,0])
plt.plot(bin_bds[2,:,1])
plt.savefig(data_path + 'bin_bds_qmi.png')

# g2bin_hist = np.histogram(bw_g2, bins='auto')
# n2bin_hist = np.histogram(bw_n2, bins='fd')
# qmibin_hist = np.histogram(bw_qmi, bins='fd')

# Create a range covering the span of each variable, spaced by the average bin width
# print(np.min(bins_g2), np.max(bins_g2), flush=True)
g2_bin_big = np.arange(np.min(bin_bds[0,:,0]), np.max(bin_bds[0,:,1]), avg_bw_g2)
n2_bin_big = np.arange(np.min(bin_bds[1,:,0]), np.max(bin_bds[1,:,1]), avg_bw_n2)
qmi_bin_big = np.arange(np.min(bin_bds[2,:,0]), np.max(bin_bds[2,:,1]), avg_bw_qmi)

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
# Calculate the probability density for each combination, and store the results
hist_shape = (len(data),)
bins = []
for arg in measures_req:
    if arg=='g2': 
        hist_shape += (len(g2_bin_big)-1,)
        bins.append(g2_bin_big)
        # print(bins, g2_bin_big)
    elif arg=='n2': 
        hist_shape += (len(n2_bin_big)-1,)
        bins.append(n2_bin_big)
    elif arg=='qmi': 
        hist_shape += (len(qmi_bin_big)-1,)
        bins.append(qmi_bin_big)

bins = np.array(bins)
hist_storage = np.zeros(hist_shape)
print(hist_shape, flush=True)
# edge_storage = np.zeros((len(data), 3, len(g2_bin_big), len(n2_bin_big), len(qmi_bin_big)))

for this_ct, data_ct in enumerate(data):
    data_requested = np.vstack([meas_in[data_ct][measure_dict[arg]] for arg in measures_req]).T

    hist3d, edges3d = calc_prob_density(data_requested, 
                                        bins)
    # hist3d, edges3d = calc_prob_density(meas_in[data_ct][0], 
    #                             meas_in[data_ct][1], 
    #                             meas_in[data_ct][2], 
    #                             g2_bin_big, n2_bin_big, qmi_bin_big)
    
    # print(hist3d.shape, edges3d[0].size, edges3d[1].size, edges3d[2].size, flush=True)
    # print(edges3d[0], edges3d)
    hist_storage[this_ct] = hist3d

    # If the histogram is empty, print that fact
    if np.all(np.isclose(hist3d,0)):
        print("Histogram is zero for data_ct {} on rank {}".format(data_ct, rank), flush=True)

    # print(hist3d)
    # edges3dp = np.array([np.array(row) for row in edges3d])
    # print(*edges3d)
    # hist_storage[this_ct,1] = np.meshgrid(edges3d[0], edges3d[1], edges3d[2])
    #
    # g2bin_min_ind = (np.min(bin_bds[0,:,0]) - bin_bds[0,data_ct,0]) // avg_bw_g2
    # new_g2bins = np.arange(np.min(bin_bds[0,:,0]) + avg_bw_g2*g2bin_min_ind, bin_bds[0,data_ct,1], avg_bw_g2)

    # n2bin_min_ind = (np.min(bin_bds[1,:,0]) - bin_bds[1,data_ct,0]) // avg_bw_n2
    # new_n2bins = np.arange(np.min(bin_bds[1,:,0]) + avg_bw_n2*n2bin_min_ind, bin_bds[1,data_ct,1], avg_bw_n2)

    # qmibin_min_ind = (np.min(bin_bds[2,:,0]) - bin_bds[2,data_ct,0]) // avg_bw_qmi
    # new_qmibins = np.arange(np.min(bin_bds[2,:,0]) + avg_bw_qmi*qmibin_min_ind, bin_bds[2,data_ct,1], avg_bw_qmi)

    # hist_storage[this_ct] = calc_prob_density(meas_in[data_ct][0], 
    #                             meas_in[data_ct][1], 
    #                             meas_in[data_ct][2], 
    #                             new_g2bins, new_n2bins, new_qmibins)

    print("Node {}, process {}".format(os.environ['SLURMD_NODENAME'], rank) \
            + " is done with job {} of {}".format(data_ct, len(combinations)), flush=True)
########################################################################################################
# comm.Barrier()


########################################################################################################
# Gather the results from all processes
if rank != 0:
    comm.Send([hist_storage, MPI.DOUBLE], dest=0)
    hist_storage2 = np.zeros((len(combinations), ) + hist_storage.shape[1:])

else:
    hist_storage2 = np.zeros((len(combinations), ) + hist_storage.shape[1:])

    for sp in range(1, size):
        hist_storage_int = np.zeros((len(to_send[sp]), ) + hist_storage.shape[1:])

        comm.Recv([hist_storage_int, MPI.DOUBLE], source=sp)

        for this_ct, data_ct in enumerate(to_send[sp]):
            hist_storage2[int(data_ct)] = hist_storage_int[this_ct]

    for this_ct, data_ct in enumerate(to_send[0]):
        hist_storage2[int(data_ct)] = hist_storage[this_ct]

########################################################################################################
comm.Barrier()
# Give all processes access to hist_storage2
for data_ct in range(len(combinations)):
    comm.Bcast([hist_storage2[data_ct], MPI.DOUBLE], root=0)
# comm.Bcast([hist_storage2, MPI.DOUBLE], root=0) # Gives an overflow when the data is over 2 GB
# print(hist_storage2.size, flush=True) # Confirmed with this print
# Have to use sends/recieves instead
# if rank == 0:
#     for sp in range(1, size):
#         comm.Send([hist_storage2, MPI.DOUBLE], dest=sp)

# else:
#     hist_storage2 = np.zeros((len(combinations), ) + hist_storage.shape[1:])
#     comm.Recv([hist_storage2, MPI.DOUBLE], source=0)
########################################################################################################
# For each energy value histogram, there might have some overlap with adjacent energy
# value histogram. Find the likelihood function for each quantity.
res_div = np.zeros((len(data),) + hist_storage.shape[1:])

# Iterate through combinations of the histograms
meas_s = hist_storage.shape[1:]
pset = powerset(meas_s)
pset = list(pset)[1:]
    
axes_to_sum = []
for i, p in enumerate(pset):
    p_tuple = tuple(p)
    
    condition = [index for index, value in enumerate(meas_s) if value not in p]
    
    axes_to_sum.append(condition)

# Find the number of arrays in condition that are not empty
lenconds = len(axes_to_sum) # - condition.count([])
# Compute also the baysian relative likelihood of the measurement for each energy.
res_comp = np.zeros((len(data), len(combinations), lenconds))



for i, data_ct in enumerate(data):
    hist_i = hist_storage[i]
    # hist_i = np.array(hist_storage2[int(data_ct)])

    denomd0 = np.sum(hist_storage2, axis=0)
    denomd = np.where(denomd0 > 1e-16, denomd0, np.full_like(hist_i, 1e-16))

    div = hist_i / denomd

    res_div[i] = div #np.sum(div)

    for this_ax, axis in enumerate(axes_to_sum):
        for j in range(len(combinations)):
            hist_j = np.array(hist_storage2[j])

            denomc0 = hist_i + hist_j
            denomc0 = np.sum(denomc0, axis=tuple(axis))
            denomc = np.where(denomc0 > 1e-16, denomc0, np.full_like(denomc0, 1e-16))

            comp = np.sum(hist_i, axis=tuple(axis)) / denomc

            # if data_ct == j: print(comp, flush=True)

            # res_comp[i, j, this_ax] = np.sum(comp) / (np.sum(hist_i, axis=tuple(axis)) > 1e-16).sum()
            res_comp[i, j, this_ax] = np.sum(comp) / (np.sum(denomc, axis=tuple(axis)) > 1e-16).sum()

    # Print the top 10 lowest values in denom0
    # print(np.sort(denom0.flatten())[:10], flush=True)
    # Print the top 10 highest values in div
    # print(np.sort(div.flatten())[-10:], flush=True)

    # div = div*avg_bw_g2

    # print(np.where(hist_j > 0))
    # where_div = np.where(hist_j > 0)
    # print(where_div)

    # if where_div.shape[0] > 0:
    # div = np.divide(hist_i, hist_j, 
    #                 out = np.full(hist_i.shape, sys.float_info.max), # sys.float_info.max
    #                 where=hist_j>0)
        # res_div[i,j] = np.sum(np.divide(hist_i, hist_j, 
        #                             out = 1.,
        #                             where=hist_j>0))
        # else: 
            # res_div[i,j] = 1.
        
    print("Histogram residual. Node {}, process {}".format(os.environ['SLURMD_NODENAME'], rank) \
            + " is done with job {} of {}".format(data_ct, len(combinations)), flush=True)

        # hist_i, edges_i = hist_storage[this_ct]
        # hist_j, edges_j = hist_storage2[j]
        # print(edges_i.shape)
        # print(inds.shape)

        # inds = np.where(np.isclose(edges_i, edges_j))
        # # Compute hist_i/hist_j on like bins
        # res_div[i,j] = np.sum(np.divide(hist_i[inds[0]], hist_j[inds[1]], 
        #                               out = 1.,
        #                               where=~np.isclose(hist_j,np.zeros_like(hist_j))))
########################################################################################################
# Gather the results from all processes
if rank != 0:
    comm.Send([res_div, MPI.DOUBLE], dest=0)
    comm.Send([res_comp, MPI.DOUBLE], dest=0)

# comm.Barrier()

if rank == 0:
    res_div2 = np.zeros((len(combinations),) + res_div.shape[1:])
    res_comp2 = np.zeros((len(combinations), len(combinations), lenconds))

    for sp in range(1, size):
        res_div_int = np.zeros((len(to_send[sp]),) + res_div.shape[1:])
        res_comp_int = np.zeros((len(to_send[sp]), len(combinations), lenconds))

        comm.Recv([res_div_int, MPI.DOUBLE], source=sp)
        comm.Recv([res_comp_int, MPI.DOUBLE], source=sp)

        for this_ct, data_ct in enumerate(to_send[sp]):
            res_div2[int(data_ct)] = res_div_int[this_ct]
            res_comp2[int(data_ct)] = res_comp_int[this_ct]

    for this_ct, data_ct in enumerate(to_send[0]):
        res_div2[int(data_ct)] = res_div[this_ct]
        res_comp2[int(data_ct)] = res_comp[this_ct]

comm.Barrier()
########################################################################################################
if rank == 0:
    # Save data
    todays_date = datetime.today().strftime('%m/%d/%Y')

    metadata = dict(ndHist_date=todays_date,
                    g2_bds=[np.min(bin_bds[0,:,0]), np.max(bin_bds[0,:,1])],
                    n2_bds=[np.min(bin_bds[1,:,0]), np.max(bin_bds[1,:,1])],
                    qmi_bds=[np.min(bin_bds[2,:,0]), np.max(bin_bds[2,:,1])])
    metadata.update(loaded_metadata)
    # metadata.update(dict(E_range=E_range, L_range=L_range, t_range=t_range,
    #                 m_measurements=m_measurements, n_times=n_times))

    # Save the simulation date as a string in format %y%m%d
    sim_date = loaded_metadata['date']
    sim_date = sim_date.strip('/')
    sim_date = sim_date.strip(r'/')
    sim_date = sim_date.replace('/', '')
    sim_date = sim_date.replace(r'/','')

    savefile = data_path + sim_date + '_nDCalcLikeHist' + '.h5'

    # h5store(savefile, g2bin_hist, 'g2bin_hist', metadata)
    # h5store(savefile, n2bin_hist, 'nbin_hist', metadata, mode='a', md_store=False)
    # h5store(savefile, qmibin_hist, 'qmibin_hist', metadata, mode='a', md_store=False)
    h5store(savefile, res_div2, 'Likelihood', metadata, mode='w') #mode='a', md_store=False)
    h5store(savefile, res_comp2, 'Comp-Likelihood', metadata, mode='a', md_store=False)
    h5store(savefile, hist_storage2, 'Histograms', metadata, mode='a', md_store=False)

if rank==0:
    print("Job completed! Time is {}".format(datetime.now()), flush=True)

comm.Barrier()
MPI.Finalize()
