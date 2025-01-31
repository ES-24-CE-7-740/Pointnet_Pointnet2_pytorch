#!/bin/bash
echo "Running combined: 400"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 16 \
	--epoch 10 \
	--gpu='0,1' \
	--log_dir 400_combined \
	--root_folder /home/agco/datasets/tractors_and_combines_combined/400/pointnet_combined

echo "Running combined: 600"
python3 train_semseg_agco.py \
        --model pointnet2_sem_seg \
        --batch_size 16 \
        --epoch 10 \
        --gpu='0,1' \
        --log_dir 600_combined \
        --root_folder /home/agco/datasets/tractors_and_combines_combined/600/pointnet_combined

echo "Running combined: 800"
python3 train_semseg_agco.py \
        --model pointnet2_sem_seg \
        --batch_size 16 \
        --epoch 10 \
        --gpu='0,1' \
        --log_dir 800_combined \
        --root_folder /home/agco/datasets/tractors_and_combines_combined/800/pointnet_combined

echo "Running combined: 1000"
python3 train_semseg_agco.py \
        --model pointnet2_sem_seg \
        --batch_size 16 \
        --epoch 10 \
        --gpu='0,1' \
        --log_dir 1000_combined \
        --root_folder /home/agco/datasets/tractors_and_combines_combined/1000/pointnet_combined

echo "Running combined - only real: 200"
python3 train_semseg_agco.py \
        --model pointnet2_sem_seg \
        --batch_size 16 \
        --epoch 10 \
        --gpu='0,1' \
        --log_dir 200_combined_only_real \
        --root_folder /home/agco/datasets/tractors_and_combines_combined/200/pointnet_only_real

echo "Running combined - only real: 400"
python3 train_semseg_agco.py \
        --model pointnet2_sem_seg \
        --batch_size 16 \
        --epoch 10 \
        --gpu='0,1' \
        --log_dir 400_combined_only_real \
        --root_folder /home/agco/datasets/tractors_and_combines_combined/400/pointnet_only_real

echo "Running combined - only real: 600"
python3 train_semseg_agco.py \
        --model pointnet2_sem_seg \
        --batch_size 16 \
        --epoch 10 \
        --gpu='0,1' \
        --log_dir 600_combined_only_real \
        --root_folder /home/agco/datasets/tractors_and_combines_combined/600/pointnet_only_real

echo "Running combined - only real: 800"
python3 train_semseg_agco.py \
        --model pointnet2_sem_seg \
        --batch_size 16 \
        --epoch 10 \
        --gpu='0,1' \
        --log_dir 800_combined_only_real \
        --root_folder /home/agco/datasets/tractors_and_combines_combined/800/pointnet_only_real

echo "Running combined - only real: 1000"
python3 train_semseg_agco.py \
        --model pointnet2_sem_seg \
        --batch_size 16 \
        --epoch 10 \
        --gpu='0,1' \
        --log_dir 1000_combined_only_real \
        --root_folder /home/agco/datasets/tractors_and_combines_combined/1000/pointnet_only_real

echo "Running combined - only real: 1200"
python3 train_semseg_agco.py \
        --model pointnet2_sem_seg \
        --batch_size 16 \
        --epoch 10 \
        --gpu='0,1' \
        --log_dir 1200_combined_only_real \
        --root_folder /home/agco/datasets/tractors_and_combines_combined/1200/pointnet_only_real
