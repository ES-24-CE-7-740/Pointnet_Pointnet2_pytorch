#!/bin/bash
#SBATCH --job-name=Zeroshot
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --output=post_exam_zeroshot_%j.out
#SBATCH --error=post_exam_zeroshot_%j.err
#SBATCH --time=100:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=morten-ks@hotmail.com

# Set the number of tasks and GPUs directly
#SBATCH --ntasks=1
#SBATCH --chdir=/home/morten/Repos/Pointnet_Pointnet2_pytorch

# Execute the job using Singularity
srun singularity exec --nv --bind /home:/home ~/containers/ptv3-container.sif \
	python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 16 \
	--epoch 32 \
	--gpu='0,1' \
	--log_dir post_exam_zeroshot \
	--root_folder /home/agco/datasets/tgch_occluded_synth
