#!/usr/bin/env bash

#SBATCH --job-name=TwoCavityPINEM_t
#SBATCH --nodes=2
#SBATCH --ntasks=20
####SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

#SBATCH --output=output.txt
#SBATCH --error=error.txt

#SBATCH --time=00:45:00

module purge
module load anaconda3
eval "$(conda shell.bash hook)"
source activate $HOME/envs/py37
# source activate /storage/home/rjm6826/envs/py37

module load mpich/4.2a1

python3 --version
which python3

which mpicc

# Print python mpi4py version
# python3 -c "from mpi4py import MPI; print(MPI.get_vendor())"

echo $LD_LIBRARY_PATH

now=$(date)
echo "$now"

srun -n $SLURM_NTASKS --mpi=pmi2 python "$@"