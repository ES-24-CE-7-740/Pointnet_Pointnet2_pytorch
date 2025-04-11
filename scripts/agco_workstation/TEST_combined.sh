echo "Testing combined: 200"
python test_semseg_agco.py \
        --log_dir 200_combined \
        --root_folder /home/agco/datasets/tractors_and_combines_combined/200/pointnet_combined

echo "Testing combined: 600"
python test_semseg_agco.py \
        --log_dir 600_combined \
        --root_folder /home/agco/datasets/tractors_and_combines_combined/600/pointnet_combined

echo "Testing combined: 800"
python test_semseg_agco.py \
        --log_dir 800_combined \
        --root_folder /home/agco/datasets/tractors_and_combines_combined/800/pointnet_combined

echo "Testing combined: 1000"
python test_semseg_agco.py \
        --log_dir 1000_combined \
        --root_folder /home/agco/datasets/tractors_and_combines_combined/1000/pointnet_combined

echo "Testing combined_only_real: 200"
python test_semseg_agco.py \
        --log_dir 200_combined_only_real \
        --root_folder /home/agco/datasets/tractors_and_combines_combined/200/pointnet_only_real

echo "Testing combined_only_real: 400"
python test_semseg_agco.py \
        --log_dir 400_combined_only_real \
        --root_folder /home/agco/datasets/tractors_and_combines_combined/400/pointnet_only_real

echo "Testing combined_only_real: 600"
python test_semseg_agco.py \
        --log_dir 600_combined_only_real \
        --root_folder /home/agco/datasets/tractors_and_combines_combined/600/pointnet_only_real

echo "Testing combined_only_real: 800"
python test_semseg_agco.py \
        --log_dir 800_combined_only_real \
        --root_folder /home/agco/datasets/tractors_and_combines_combined/800/pointnet_only_real

echo "Testing combined_only_real: 1000"
python test_semseg_agco.py \
        --log_dir 1000_combined_only_real \
        --root_folder /home/agco/datasets/tractors_and_combines_combined/1000/pointnet_only_real

echo "Testing combined_only_real: 1200"
python test_semseg_agco.py \
        --log_dir 1200_combined_only_real \
        --root_folder /home/agco/datasets/tractors_and_combines_combined/1200/pointnet_only_real
