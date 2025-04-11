#!/bin/bash
echo "Training blue valtra:"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 32 \
	--epoch 32 \
	--gpu='0' \
	--log_dir ucloud_blue_valtra \
	--root_folder /work/3dgs-drive/data/agco_zs_synth/blue_valtra

echo "Testing blue valtra"
python test_semseg_agco.py \
        --log_dir ucloud_blue_valtra \
        --root_folder /work/3dgs-drive/data/agco_zs_synth/blue_valtra
