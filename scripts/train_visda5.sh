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
python train_tar_new5.py --loss_type dot --loss_wt nn0 --data_trans bs --data_aug 0.2,0.5 --bn_adapt 0 --lp_type 0.5 --div_wt 0.00 --fuse_af 5 --fuse_type m --debug --exp_name unim_nn0_dot_bs_lp05_af5m
"
