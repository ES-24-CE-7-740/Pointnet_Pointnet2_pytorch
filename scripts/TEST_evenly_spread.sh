echo "Testing evenly_spread: 200"
python test_semseg_agco.py \
        --log_dir evenly_sampled_200 \
        --root_folder /home/agco/datasets/tractors_and_combines_combined_even_sampling/200

echo "Testing evenly_spread: 400"
python test_semseg_agco.py \
        --log_dir evenly_sampled_400 \
        --root_folder /home/agco/datasets/tractors_and_combines_combined_even_sampling/400

echo "Testing evenly_spread: 600"
python test_semseg_agco.py \
        --log_dir evenly_sampled_600 \
        --root_folder /home/agco/datasets/tractors_and_combines_combined_even_sampling/600

echo "Testing evenly_spread: 800"
python test_semseg_agco.py \
        --log_dir evenly_sampled_800 \
        --root_folder /home/agco/datasets/tractors_and_combines_combined_even_sampling/800

echo "Testing evenly_spread: 1000"
python test_semseg_agco.py \
        --log_dir evenly_sampled_1000 \
        --root_folder /home/agco/datasets/tractors_and_combines_combined_even_sampling/1000

echo "Testing evenly_spread: 1200"
python test_semseg_agco.py \
        --log_dir evenly_sampled_1200 \
        --root_folder /home/agco/datasets/tractors_and_combines_combined_even_sampling/1200