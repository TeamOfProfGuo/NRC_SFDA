#!/bin/bash

#SBATCH --job-name=pre_visda
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
python train_tar_new7.py --loss_type dot --loss_wt en5 --data_trans mn --data_aug 0.2,0.5 --lp_type 0.5 --fuse_af 10 --fuse_type m --debug --k 25 --exp_name unim_en5_dot_mn2_lp05_af10m_k25
"
