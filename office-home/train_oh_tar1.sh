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
WT=$1
AF=$2
FT=$3

# Singularity path
ext3_path=/scratch/$USER/python36/python36.ext3
sif_path=/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif


# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
python -m office-home.train_tar_new1 --dset a2c  --loss_wt ${WT} --lp_type 0.5 --data_trans mn --data_aug 0.2,0.5 --div_wt 0.1 --fuse_af ${AF} --k 9 --fuse_type ${FT} --lr_scale 0.5 --debug --exp_name dot_${WT}_mn2_lp05_div01_af${AF}${FT}
python -m office-home.train_tar_new1 --dset a2p  --loss_wt ${WT} --lp_type 0.5 --data_trans mn --data_aug 0.2,0.5 --div_wt 0.1 --fuse_af ${AF} --k 9 --fuse_type ${FT} --lr_scale 0.5 --debug --exp_name dot_${WT}_mn2_lp05_div01_af${AF}${FT}
python -m office-home.train_tar_new1 --dset a2r  --loss_wt ${WT} --lp_type 0.5 --data_trans mn --data_aug 0.2,0.5 --div_wt 0.1 --fuse_af ${AF} --k 9 --fuse_type ${FT} --lr_scale 0.5 --debug --exp_name dot_${WT}_mn2_lp05_div01_af${AF}${FT}

python -m office-home.train_tar_new1 --dset c2a  --loss_wt ${WT} --lp_type 0.5 --data_trans mn --data_aug 0.2,0.5 --div_wt 0.1 --fuse_af ${AF} --k 9 --fuse_type ${FT} --lr_scale 0.5 --debug --exp_name dot_${WT}_mn2_lp05_div01_af${AF}${FT}
python -m office-home.train_tar_new1 --dset c2p  --loss_wt ${WT} --lp_type 0.5 --data_trans mn --data_aug 0.2,0.5 --div_wt 0.1 --fuse_af ${AF} --k 9 --fuse_type ${FT} --lr_scale 0.5 --debug --exp_name dot_${WT}_mn2_lp05_div01_af${AF}${FT}
python -m office-home.train_tar_new1 --dset c2r  --loss_wt ${WT} --lp_type 0.5 --data_trans mn --data_aug 0.2,0.5 --div_wt 0.1 --fuse_af ${AF} --k 9 --fuse_type ${FT} --lr_scale 0.5 --debug --exp_name dot_${WT}_mn2_lp05_div01_af${AF}${FT}

python -m office-home.train_tar_new1 --dset p2a  --loss_wt ${WT} --lp_type 0.5 --data_trans mn --data_aug 0.2,0.5 --div_wt 0.1 --fuse_af ${AF} --k 9 --fuse_type ${FT} --lr_scale 0.5 --debug --exp_name dot_${WT}_mn2_lp05_div01_af${AF}${FT}
python -m office-home.train_tar_new1 --dset p2c  --loss_wt ${WT} --lp_type 0.5 --data_trans mn --data_aug 0.2,0.5 --div_wt 0.1 --fuse_af ${AF} --k 9 --fuse_type ${FT} --lr_scale 0.5 --debug --exp_name dot_${WT}_mn2_lp05_div01_af${AF}${FT}
python -m office-home.train_tar_new1 --dset p2r  --loss_wt ${WT} --lp_type 0.5 --data_trans mn --data_aug 0.2,0.5 --div_wt 0.1 --fuse_af ${AF} --k 9 --fuse_type ${FT} --lr_scale 0.5 --debug --exp_name dot_${WT}_mn2_lp05_div01_af${AF}${FT}

python -m office-home.train_tar_new1 --dset r2a  --loss_wt ${WT} --lp_type 0.5 --data_trans mn --data_aug 0.2,0.5 --div_wt 0.1 --fuse_af ${AF} --k 9 --fuse_type ${FT} --lr_scale 0.5 --debug --exp_name dot_${WT}_mn2_lp05_div01_af${AF}${FT}
python -m office-home.train_tar_new1 --dset r2c  --loss_wt ${WT} --lp_type 0.5 --data_trans mn --data_aug 0.2,0.5 --div_wt 0.1 --fuse_af ${AF} --k 9 --fuse_type ${FT} --lr_scale 0.5 --debug --exp_name dot_${WT}_mn2_lp05_div01_af${AF}${FT}
python -m office-home.train_tar_new1 --dset r2p  --loss_wt ${WT} --lp_type 0.5 --data_trans mn --data_aug 0.2,0.5 --div_wt 0.1 --fuse_af ${AF} --k 9 --fuse_type ${FT} --lr_scale 0.5 --debug --exp_name dot_${WT}_mn2_lp05_div01_af${AF}${FT}
"

# --plabel_soft


# lp_type = 1.0 : label propagation using soft labels

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