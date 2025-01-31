#!/bin/bash
echo "\n\nTraining evenly spread: 200"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 16 \
	--epoch 10 \
	--gpu='0,1' \
	--log_dir evenly_sampled_200 \
	--root_folder /home/agco/datasets/tractors_and_combines_combined_even_sampling/200

echo "\n\nTraining evenly spread: 400"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 16 \
	--epoch 10 \
	--gpu='0,1' \
	--log_dir evenly_sampled_400 \
	--root_folder /home/agco/datasets/tractors_and_combines_combined_even_sampling/400

echo "\n\nTraining evenly spread: 600"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 16 \
	--epoch 10 \
	--gpu='0,1' \
	--log_dir evenly_sampled_600 \
	--root_folder /home/agco/datasets/tractors_and_combines_combined_even_sampling/600

echo "\n\nTraining evenly spread: 800"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 16 \
	--epoch 10 \
	--gpu='0,1' \
	--log_dir evenly_sampled_800 \
	--root_folder /home/agco/datasets/tractors_and_combines_combined_even_sampling/800

echo "\n\nTraining evenly spread: 1000"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 16 \
	--epoch 10 \
	--gpu='0,1' \
	--log_dir evenly_sampled_1000 \
	--root_folder /home/agco/datasets/tractors_and_combines_combined_even_sampling/1000

echo "\n\nTraining evenly spread: 1200"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 16 \
	--epoch 10 \
	--gpu='0,1' \
	--log_dir evenly_sampled_1200 \
	--root_folder /home/agco/datasets/tractors_and_combines_combined_even_sampling/1200