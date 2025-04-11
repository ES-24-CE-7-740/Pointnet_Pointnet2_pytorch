#!/bin/bash
echo "Training mixed - 100%:"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 32 \
	--epoch 32 \
	--gpu='0' \
	--log_dir ucloud_mixed_100 \
	--root_folder /work/3dgs-drive/data/agco_all_real_only/5000/mixed_100

echo "Testing mixed - 100%"
python test_semseg_agco.py \
	--log_dir ucloud_mixed_100 \
	--root_folder /work/3dgs-drive/data/agco_zs_synth/blue_valtra


echo "Training real - 100%:"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 32 \
	--epoch 32 \
	--gpu='0' \
	--log_dir ucloud_real_100 \
	--root_folder /work/3dgs-drive/data/agco_all_real_only/5000/real_100

echo "Testing mixed - 100%"
python test_semseg_agco.py \
	--log_dir ucloud_real_100 \
	--root_folder /work/3dgs-drive/data/agco_zs_synth/blue_valtra