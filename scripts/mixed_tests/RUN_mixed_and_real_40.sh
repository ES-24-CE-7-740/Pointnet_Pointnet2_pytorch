#!/bin/bash
echo "Training mixed - 40%:"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 32 \
	--epoch 32 \
	--gpu='0' \
	--log_dir ucloud_mixed_40 \
	--root_folder /work/3dgs-drive/data/agco_all_real_only/2000/mixed_40

echo "Testing mixed - 40%"
python test_semseg_agco.py \
	--log_dir ucloud_mixed_40 \
	--root_folder /work/3dgs-drive/data/agco_zs_synth/blue_valtra


echo "Training real - 40%:"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 32 \
	--epoch 32 \
	--gpu='0' \
	--log_dir ucloud_real_40 \
	--root_folder /work/3dgs-drive/data/agco_all_real_only/2000/real_40

echo "Testing mixed - 40%"
python test_semseg_agco.py \
	--log_dir ucloud_real_40 \
	--root_folder /work/3dgs-drive/data/agco_zs_synth/blue_valtra