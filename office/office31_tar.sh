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
python -m office.office31_tar --dset a2d --loss_type dot --loss_wt en5  --exp_name unim_dot_en5
python -m office.office31_tar --dset a2w --loss_type dot --loss_wt en5  --exp_name unim_dot_en5

python -m office.office31_tar --dset d2a --loss_type dot --loss_wt en5  --exp_name unim_dot_en5
python -m office.office31_tar --dset d2w --loss_type dot --loss_wt en5  --exp_name unim_dot_en5

python -m office.office31_tar --dset w2a --loss_type dot --loss_wt en5  --exp_name unim_dot_en5
python -m office.office31_tar --dset w2d --loss_type dot --loss_wt en5  --exp_name unim_dot_en5
"


# python office31_tar.py --dset a2d  --K 3 --beta 1  --file k3b1

# python office31_tar.py --dset w2a  --K 3  --beta 1  --file k3b1

# python office31_tar.py --dset a2w  --K 3   --beta 1  --file k3b1

# python office31_tar.py --dset d2w  --K 3   --beta 1  --file k3b1

# python office31_tar.py --dset d2a --K 3 --beta 1  --file k3b1

# python office31_tar.py --dset w2d  --K 3   --beta 1  --file k3b1