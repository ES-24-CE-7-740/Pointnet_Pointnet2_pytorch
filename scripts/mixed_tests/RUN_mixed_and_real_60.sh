#!/bin/bash
echo "Training mixed - 60%:"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 32 \
	--epoch 32 \
	--gpu='0' \
	--log_dir ucloud_mixed_60 \
	--root_folder /work/3dgs-drive/data/agco_all_real_only/3000/mixed_60

echo "Testing mixed - 60%"
python test_semseg_agco.py \
	--log_dir ucloud_mixed_60 \
	--root_folder /work/3dgs-drive/data/agco_zs_synth/blue_valtra


echo "Training real - 60%:"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 32 \
	--epoch 32 \
	--gpu='0' \
	--log_dir ucloud_real_60 \
	--root_folder /work/3dgs-drive/data/agco_all_real_only/3000/real_60

echo "Testing mixed - 60%"
python test_semseg_agco.py \
	--log_dir ucloud_real_60 \
	--root_folder /work/3dgs-drive/data/agco_zs_synth/blue_valtra