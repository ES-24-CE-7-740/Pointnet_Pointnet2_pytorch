#!/bin/bash
#SBATCH --job-name=pointnet_test
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --output=post_exam_mixed_test_%j.out
#SBATCH --error=post_exam_mixed_test_%j.err

# Set the number of tasks and GPUs directly
#SBATCH --ntasks=1
#SBATCH --chdir=/home/morten/Repos/Pointnet_Pointnet2_pytorch

# Execute the job using Singularity
srun singularity exec --nv --bind /home:/home ~/containers/ptv3-container.sif \
	python test_semseg_agco.py --log_dir post_exam_combined
