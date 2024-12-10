#!/bin/bash
#SBATCH --job-name=pointnet_train
#SBATCH --cpus-per-task=40
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=pointnet_train_%j.out
#SBATCH --error=pointnet_train_%j.err

# Set the number of tasks and GPUs directly
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2

# Execute the job using Singularity
srun singularity exec --nv --bind /ceph:/ceph ~/custom_containers/pytorch_22.09-py3.sif \
	python3 /ceph/project/ce-7-740/project/repos/Pointnet_Pointnet2_pytorch/train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 16 \
	--epoch 10 \
	--gpu='5,6' \
	--log_dir pointnet2_synth \
	--root_folder /ceph/project/ce-7-740/project/repos/Pointnet_Pointnet2_pytorch/data
