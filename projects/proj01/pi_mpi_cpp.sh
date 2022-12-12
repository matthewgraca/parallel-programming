#!/bin/bash
#SBATCH --job-name=PI_MPI       # job name
#SBATCH --output=PI_MPI_%j.log  # log file name
#SBATCH --partition=compute     # use comupting cluster
#SBATCH --mem=1gb               # job mem request
#SBATCH --nodes=4               # number of comupting nodes
#SBATCH --time=00:02:00         # time limit HH:MM:SS

. /etc/profile.d/modules.sh

module load openmpi/2.1.2

/opt/openmpi-2.1.2/bin/mpirun ./pi_mpi
