#!/bin/bash
echo "Training fendt300:"
python3 train_semseg_agco.py \
	--model pointnet2_sem_seg \
	--batch_size 32 \
	--epoch 32 \
	--gpu='0' \
	--log_dir ucloud_fendt300 \
	--root_folder /work/3dgs-drive/data/agco_zs_synth/fendt300

echo "Testing fendt300"
python test_semseg_agco.py \
        --log_dir ucloud_fendt300 \
        --root_folder /work/3dgs-drive/data/agco_zs_synth/blue_valtra
