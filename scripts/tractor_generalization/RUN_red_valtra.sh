#!/bin/bash
echo "Training red valtra:"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 32 \
	--epoch 32 \
	--gpu='0' \
	--log_dir ucloud_red_valtra \
	--root_folder /work/3dgs-drive/data/agco_zs_synth/red_valtra

echo "Testing red valtra"
python test_semseg_agco.py \
        --log_dir ucloud_red_valtra \
        --root_folder /work/3dgs-drive/data/agco_zs_synth/blue_valtra
