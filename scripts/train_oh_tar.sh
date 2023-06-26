#!/bin/bash

#SBATCH --job-name=LP_BN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000

# job info

# Singularity path
ext3_path=/scratch/$USER/python36/python36.ext3
sif_path=/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif


# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
python -m office-home.train_tar_new 
"



# python train_tar.py --home --dset a2r  --K 3 --KK 2 --file target --gpu_id 0

# python train_tar.py --home --dset r2a  --K 3 --KK 2 --file target --gpu_id 0

# python train_tar.py --home --dset r2c  --K 3 --KK 2 --file target --gpu_id 0

# python train_tar.py --home --dset r2p  --K 3 --KK 2 --file target --gpu_id 0

# python train_tar.py --home --dset p2a  --K 3 --KK 2 --file target --gpu_id 0

# python train_tar.py --home --dset p2c  --K 3 --KK 2 --file target --gpu_id 0

# python train_tar.py --home --dset a2p  --K 3 --KK 2 --file target --gpu_id 0

# python train_tar.py --home --dset a2c  --K 3 --KK 2 --file target --gpu_id 0

# python train_tar.py --home --dset p2r  --K 3 --KK 2 --file target --gpu_id 0

# python train_tar.py --home --dset c2a  --K 3 --KK 2 --file target --gpu_id 0

# python train_tar.py --home --dset c2p  --K 3 --KK 2 --file target --gpu_id 0

# python train_tar.py --home --dset c2r  --K 3 --KK 2 --file target --gpu_id 0