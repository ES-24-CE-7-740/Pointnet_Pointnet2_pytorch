import os
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path

def normalize_pc(points):
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
    points /= furthest_distance

    return points

def process_tractors_and_combines(root, num_points):
    # Load data
    root = Path(root)

    train_seq = ['011']
    val_seq = ['111']
    test_seq = ['3000', '3001', '3002', '3005', '3007']

    # Initialize lists for points and labels
    train_points = []
    train_labels = []
    validate_points = []
    validate_labels = []
    test_points = []
    test_labels = []

    # Fetch training points
    for seq in train_seq:
        sequence_path = root / 'dataset' / 'sets' / seq
        points_path = sequence_path / 'points'
        labels_path = sequence_path / 'labels'

        # Add points and labels to the respective lists
        train_points.extend(points_path.iterdir())
        train_labels.extend(labels_path.iterdir())

    # Fetch validation points
    for seq in val_seq:
        sequence_path = root / 'dataset' / 'sets' / seq
        points_path = sequence_path / 'points'
        labels_path = sequence_path / 'labels'

        # Add points and labels to the respective lists
        validate_points.extend(points_path.iterdir())
        validate_labels.extend(labels_path.iterdir())

    # Fetch test points
    for seq in test_seq:
        sequence_path = root / 'dataset' / 'sets' / seq
        points_path = sequence_path / 'points'
        labels_path = sequence_path / 'labels'

        # Add points and labels to the respective lists
        test_points.extend(points_path.iterdir())
        test_labels.extend(labels_path.iterdir())

    # Sort the points and labels
    train_points = sorted(train_points)
    train_labels = sorted(train_labels)
    validate_points = sorted(validate_points)
    validate_labels = sorted(validate_labels)
    test_points = sorted(test_points)
    test_labels = sorted(test_labels)

    splits_str = ['train', 'validate', 'test']
    splits_data = [train_points, validate_points, test_points]
    splits_labels = [train_labels, validate_labels, test_labels]
    
    # Get the label mapping dictionary.
    label_map = get_learning_map(-1)

    # Process data
    # - Add normalized rgb values to the pointcloud (0.5, 0.5, 0.5)
    # - Add normalized xyz values to the pointcloud (Normalized to the unit sphere -> [-1, 1])
    print(f'Processing data at: "{root}"')
    print(f'Sampling point clouds with {num_points} points...')
    for split_name, split_data, split_label in zip(splits_str, splits_data, splits_labels):
        
        # Pathing of processed data
        addi_path = "blue_valtra"

        save_dir = os.path.join(root, addi_path, 'processed_pointnet2', f'{split_name}')
        points_dir = os.path.join(save_dir, 'points')
        labels_dir = os.path.join(save_dir, 'labels')
        
        # Create directories
        try: 
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(points_dir, exist_ok=False)
            os.makedirs(labels_dir, exist_ok=False)
        except OSError as e: print(e); exit(1)
        
        # Process each pointcloud
        for data_fn, label_fn in tqdm(zip(split_data, split_label), total=len(split_data), desc=f'Processing {split_name} data'):
            # Load the pointcloud and label
            pointcloud = np.load(data_fn)
            label = np.load(label_fn)
            # Only keep the xyz coordinates
            pointcloud = pointcloud[:, :3]
            
            # Ensure pointcloud size is consistent with num_points
            # If the number of points in the data is less than `num_points`, sample with replacement for missing points
            if pointcloud.shape[0] < num_points:
                # Use all points first
                full_choice = np.arange(pointcloud.shape[0])
                
                # Randomly sample additional points to make up the difference
                additional_choice = np.random.choice(pointcloud.shape[0], num_points - pointcloud.shape[0], replace=True)
                
                # Combine the indices
                choice = np.concatenate([full_choice, additional_choice])

            # If the number of points in the data is greater than `num_points`, sample without replacement
            else: 
                choice = np.random.choice(pointcloud.shape[0], num_points, replace=False)
            
            pointcloud = pointcloud[choice, :]
            label = label[choice]
            
            # Create normalized rgb channels
            normalized_rgb = np.full_like(pointcloud, 0.5, dtype=np.float32)
            
            # Create normalized xyz channels
            normalized_pc = normalize_pc(pointcloud)
            
            # Concatenate the normalized rgb and xyz channels
            pointcloud_processed = np.concatenate((pointcloud, normalized_rgb, normalized_pc), axis=1)
            
            # Convert the pointcloud to float16
            pointcloud_processed = pointcloud_processed.astype(np.float16)
            
            # Map individual vehicle label to general labels
            mapped_labels = np.vectorize(label_map.get)(label)

            # Convert the label to int8
            label = mapped_labels.round().astype(np.int8)
            

            # Fetch the sequence
            data_fn = Path(data_fn)
            sequence = data_fn.parts[-3]
            
            # Save the processed pointcloud and label
            np.save(os.path.join(save_dir, 'points', sequence + os.path.basename(data_fn)), pointcloud_processed)
            np.save(os.path.join(save_dir, 'labels', sequence + os.path.basename(label_fn)), label)
    
    # Save the number of points sampled
    with open(os.path.join(root, addi_path, 'processed_pointnet2', 'num_points.txt'), 'w') as file:
        file.write(str(num_points))

    print('Processing complete!')

def get_learning_map(ignore_index):
    learning_map = {
        ignore_index: ignore_index,
        41:2, # katana (forage harvester)
        20:2, # combine
        21:2, # ideal_10t (combine harvester)
        22:2, # fendt_paralevel (combine harvester)
        23:2, # laverda (combine harvester)
        10:1, # tractor
        11:1, # blue_valtra (tractor)
        12:1, # grey_valtra (tractor)
        13:1, # massey (tractor)
        14:1, # fendt300 (tractor)
        15:1, # fendt1000 (tractor)
        16:1, # orange_valtra (tractor)
        17:1, # red_valtra (tractor)
        18:1, # new_holland (tractor - not in real)
        19:1, # deer_kramer (tractor - not in real)
        30:0, # trailer
        0:0,
        1:0,
    }
    return learning_map
        

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Process Tractors and Combines dataset')
    # parser.add_argument('--root', type=str, default='data/', 
    #                     help='Path to the root directory of the dataset')
    
    # parser.add_argument('--num_points', type=int, default=30000, 
    #                     help='Number of points to sample from the pointcloud')

    
    # args = parser.parse_args()
    
    # # Process the dataset
    # process_tractors_and_combines(root=args.root, num_points=args.num_points)
    
    # For debugging
    process_tractors_and_combines(root='/work/3dgs-drive/data/agco_zs_synth/', num_points=30000)

