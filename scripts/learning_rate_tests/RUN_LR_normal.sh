#!/bin/bash
echo "Running PointNet normal LR Test:"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 32 \
	--epoch 20 \
	--learning_rate 0.002 \
	--gpu='0' \
	--log_dir ucloud_LR_normal \
	--root_folder /work/3dgs-drive/data/tractors_and_combines_ablation/pointnet_ablation

echo "Testing normal LR"
python test_semseg_agco.py \
        --log_dir ucloud_LR_normal \
        --root_folder /work/3dgs-drive/data/tractors_and_combines_ablation/pointnet_ablation
