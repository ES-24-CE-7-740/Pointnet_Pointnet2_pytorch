#!/bin/bash
echo "Training mixed - 20%:"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 32 \
	--epoch 32 \
	--gpu='0' \
	--log_dir ucloud_mixed_20 \
	--root_folder /work/3dgs-drive/data/agco_all_real_only/1000/mixed_20

echo "Testing mixed - 20%"
python test_semseg_agco.py \
	--log_dir ucloud_mixed_20 \
	--root_folder /work/3dgs-drive/data/agco_zs_synth/blue_valtra


echo "Training real - 20%:"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 32 \
	--epoch 32 \
	--gpu='0' \
	--log_dir ucloud_real_20 \
	--root_folder /work/3dgs-drive/data/agco_all_real_only/1000/real_20

echo "Testing mixed - 20%"
python test_semseg_agco.py \
	--log_dir ucloud_real_20 \
	--root_folder /work/3dgs-drive/data/agco_zs_synth/blue_valtra