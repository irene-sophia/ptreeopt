#!/bin/bash
#SBATCH -D /home/jdherman/ptree
#SBATCH -o /home/jdherman/ptree/job.%j.%N.out
#SBATCH -e /home/jdherman/ptree/job.%j.%N.err
#SBATCH -n 97            # Total number of processors to request (32 cores per node)
#SBATCH -p med           # Queue name hi/med/lo
#SBATCH -t 70:00:00        # Run time (hh:mm:ss) - 24 hours
#SBATCH --mail-user=jdherman@ucdavis.edu              # address for email notification
#SBATCH --mail-type=ALL                  # email at Begin and End of job

# module load intel openmpi
mpirun -n 97 python main-parallel-cc.py
