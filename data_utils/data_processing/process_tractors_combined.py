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

def process_tractors_and_combines(root, num_points, only_real=False):
    # Load data
    root = Path(root)
    if only_real != True:
        sequences = ['00', '01', '10'] # combined
    else:
        sequences = ['00', '01'] # Only real

    # Initialize lists for points and labels
    train_points = []
    train_labels = []

    for seq in sequences:
        sequence_path = root / 'dataset' / 'sets' / seq
        points_path = sequence_path / 'points'
        labels_path = sequence_path / 'labels'

        # Add points and labels to the respective lists
        train_points.extend(points_path.iterdir())
        train_labels.extend(labels_path.iterdir())

    validate_path = os.path.join(root, 'dataset/sets/03')
    validate_data = [os.path.join(validate_path, 'points', f) for f in os.listdir(os.path.join(validate_path, 'points'))]
    validate_labels = [os.path.join(validate_path, 'labels', f) for f in os.listdir(os.path.join(validate_path, 'labels'))]
    
    test_path = os.path.join(root, 'dataset/sets/02')
    test_data = [os.path.join(test_path, 'points', f) for f in os.listdir(os.path.join(test_path, 'points'))]
    test_labels = [os.path.join(test_path, 'labels', f) for f in os.listdir(os.path.join(test_path, 'labels'))]
    
    splits_str = ['train', 'validate', 'test']
    splits_data = [train_points, validate_data, test_data]
    splits_labels = [train_labels, validate_labels, test_labels]
    
    # Process data
    # - Add normalized rgb values to the pointcloud (0.5, 0.5, 0.5)
    # - Add normalized xyz values to the pointcloud (Normalized to the unit sphere -> [-1, 1])
    print(f'Processing data at: "{root}"')
    print(f'Sampling point clouds with {num_points} points...')
    for split_name, split_data, split_label in zip(splits_str, splits_data, splits_labels):
        
        # Pathing of processed data
        if only_real:
            addi_path = 'pointnet_only_real'
        else:
            addi_path = 'pointnet_combined'

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
            
            # Convert the label to int8
            label = label.round().astype(np.int8)
            
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
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Tractors and Combines dataset')
    parser.add_argument('--root', type=str, default='data/', 
                        help='Path to the root directory of the dataset')
    
    parser.add_argument('--num_points', type=int, default=30000, 
                        help='Number of points to sample from the pointcloud')

    parser.add_argument('--only_real', type=bool, default=False, help='If you want to train only on upsampled real data.')
    
    args = parser.parse_args()
    
    # Process the dataset
    process_tractors_and_combines(root=args.root, num_points=args.num_points, only_real=args.only_real)
