#!/bin/bash
echo 'Starting reduced real dataset training!'

# Create a directory for logs if it doesn't exist
mkdir -p log/reduced_logs

run_training() {
    echo "Running: $1"
    eval "$2 >> log/reduced_logs/$3.log 2>&1"
    if [ $? -ne 0 ]; then
        echo "Error: Training step failed: $1 (check logs/$3.log for details)"
        echo "Continuing with the next training job..."
        #exit 1  # Stop script on failure
    fi
    echo "Completed: $1 (logs saved to logs/$3.log)"
}

run_training "Training on 200 pcs for 300 epochs..." \
    "CUDA_VISIBLE_DEVICE=0,1 python train_semseg_agco.py --model pointnet2_sem_seg --epoch 300 --root_folder '/home/agco/datasets/agco_real/reduced/200' --log_dir 'size_200_test_300_epochs'" \
    "training_200_300"

run_training "Training on 400 pcs for 150 epochs..." \
    "CUDA_VISIBLE_DEVICE=0,1 python train_semseg_agco.py --model pointnet2_sem_seg --epoch 150 --root_folder '/home/agco/datasets/agco_real/reduced/400' --log_dir 'size_400_test_150_epochs'" \
    "training_400_150"

run_training "Training on 600 pcs for 100 epochs..." \
    "CUDA_VISIBLE_DEVICE=0,1 python train_semseg_agco.py --model pointnet2_sem_seg --epoch 100 --root_folder '/home/agco/datasets/agco_real/reduced/600' --log_dir 'size_600_test_100_epochs'" \
    "training_600_100"

run_training "Training on 800 pcs for 75 epochs..." \
    "CUDA_VISIBLE_DEVICE=0,1 python train_semseg_agco.py --model pointnet2_sem_seg --epoch 75 --root_folder '/home/agco/datasets/agco_real/reduced/800' --log_dir 'size_800_test_75_epochs'" \
    "training_800_75"

run_training "Training on 1000 pcs for 60 epochs..." \
    "CUDA_VISIBLE_DEVICE=0,1 python train_semseg_agco.py --model pointnet2_sem_seg --epoch 60 --root_folder '/home/agco/datasets/agco_real/reduced/1000' --log_dir 'size_1000_test_60_epochs'" \
    "training_1000_60"

echo 'Training completed!'
