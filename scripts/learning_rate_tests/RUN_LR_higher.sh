#!/bin/bash
echo "Running PointNet higher LR Test:"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 32 \
	--epoch 20 \
	--learning_rate 0.02 \
	--gpu='0' \
	--log_dir ucloud_LR_higher \
	--root_folder /work/3dgs-drive/data/tractors_and_combines_ablation/pointnet_ablation

echo "Testing higher LR"
python test_semseg_agco.py \
        --log_dir ucloud_LR_higher \
        --root_folder /work/3dgs-drive/data/tractors_and_combines_ablation/pointnet_ablation
