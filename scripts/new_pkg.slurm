#!/bin/bash

# Singularity path
ext3_path=/scratch/$USER/python36/python36.ext3
sif_path=/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif

# start running
singularity exec --nv --bind /scratch/$USER \
--overlay /scratch/$USER/python36/python36.ext3:rw \
/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
source /ext3/env.sh
conda install faiss-gpu -c pytorch
"
