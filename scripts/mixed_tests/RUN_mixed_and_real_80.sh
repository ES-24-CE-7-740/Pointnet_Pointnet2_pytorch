#!/bin/bash
echo "Training mixed - 80%:"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 32 \
	--epoch 32 \
	--gpu='0' \
	--log_dir ucloud_mixed_80 \
	--root_folder /work/3dgs-drive/data/agco_all_real_only/4000/mixed_80

echo "Testing mixed - 80%"
python test_semseg_agco.py \
	--log_dir ucloud_mixed_80 \
	--root_folder /work/3dgs-drive/data/agco_zs_synth/blue_valtra


echo "Training real - 80%:"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 32 \
	--epoch 32 \
	--gpu='0' \
	--log_dir ucloud_real_80 \
	--root_folder /work/3dgs-drive/data/agco_all_real_only/4000/real_80

echo "Testing mixed - 80%"
python test_semseg_agco.py \
	--log_dir ucloud_real_80 \
	--root_folder /work/3dgs-drive/data/agco_zs_synth/blue_valtra