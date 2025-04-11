#!/bin/bash
echo "Running PointNet Test:"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 32 \
	--epoch 20 \
	--gpu='0' \
	--log_dir ucloud_epochs_20 \
	--root_folder /work/3dgs-drive/data/tractors_and_combines_ablation/pointnet_ablation


echo "Testing epoch 20"
python test_semseg_agco.py \
        --log_dir ucloud_epochs_20 \
        --root_folder /work/3dgs-drive/data/tractors_and_combines_ablation/pointnet_ablation
