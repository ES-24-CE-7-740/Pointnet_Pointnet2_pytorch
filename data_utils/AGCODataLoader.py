from pathlib import Path
import os
import sys
import glob
import numpy as np
import torch
import json
import pickle
from torch.utils.data import Dataset

# Specs:
#   pointcloud.shape == (num_points, 3)
#   seg.shape == (num_points,)
class TractorsAndCombines(Dataset):
    def __init__(self, root, split='train', augment=False):
        self.root = Path(root)
        self.split = split
        self.augment = augment # Enable/disable data augmentation
        self.sem_classes = {'Ground': [0], 'Tractor': [1], 'Combine Harvester': [2]}
        self.processed_dir = self.root.joinpath('processed_pointnet2')
        
        if self.processed_dir.exists():
            print('Processed data found!')
        else:
            raise RuntimeError('Processed data not found! Please run "prepare_tractors.py" first!')
        
        # Get the pointcloud and label file paths and sort them
        if split == 'train' or split == 'validate' or split == 'test':
            self.pointcloud_files = sorted([self.processed_dir.joinpath(split, 'points', f) 
                                            for f in (self.processed_dir.joinpath(split, 'points')).iterdir()])
            self.label_files = sorted([self.processed_dir.joinpath(split, 'labels', f)
                                       for f in (self.processed_dir.joinpath(split, 'labels')).iterdir()])
        else:
            print(f'Invalid split: "{self.split}"! Please use "train", "validate", or "test"!')
            sys.exit(1)
        
        # Calculate label weights
        if split=='train':
            class_counts = np.zeros(3)

            for label_file in sorted([file for file in self.processed_dir.joinpath(split, 'labels').iterdir()]):
                label = np.load(label_file)
                for class_id in range(3):
                    class_counts[class_id] += np.sum(label == class_id)

            total_points = np.sum(class_counts)
            class_frequencies = class_counts / total_points

            self.labelweights = 1 / (class_frequencies + 1e-6)

            # Normalize weights
            self.labelweights /= np.sum(self.labelweights)
        
    # Set some class properties
    @property
    def classes(self):
        return self.sem_classes
    
    @property
    def num_classes(self):
        return len(self.sem_classes)
    
    @property
    def num_points(self):
        return np.load(self.pointcloud_files[0]).shape[0]
    
    @property
    def num_features(self):
        return np.load(self.pointcloud_files[0]).shape[1]
    
    
    def __getitem__(self, index):
        # Load the pointcloud
        pointcloud: np.ndarray = np.load(self.pointcloud_files[index]).astype(np.float32) # Should be saved as np.float16
        
        # Load the label
        seg: np.ndarray = np.load(self.label_files[index]) # Should be saved as np.int8
        
        # Make sure the label is one of the three classes
        seg = np.where(seg > 2, 0, seg)

        # Apply data augmentation for the 'train' split only
        if self.augment and self.split == 'train':
            pointcloud, seg = self.augment_pointcloud(pointcloud, seg)
        
        pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
        seg = torch.tensor(seg, dtype=torch.int64)
        seg = torch.squeeze(input=seg) # Ensure shape is [num_points] instead of [num_points, 1]
        
        return pointcloud, seg
    
    def __len__(self):
        return len(self.pointcloud_files)
    
    def augment_pointcloud(self, pointcloud, seg):
        """Apply augmentations to the point cloud and corresponding labels."""
        # Random rotation around z-axis
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        pointcloud[:, :3] = pointcloud[:, :3] @ rotation_matrix.T

        # Add Gaussian noise
        noise = np.random.normal(0, 0.01, size=pointcloud.shape)
        pointcloud += noise

        # Random scaling
        scale = np.random.uniform(0.8, 1.2)
        pointcloud[:, :3] *= scale

        # Random translation
        translation = np.random.uniform(-0.1, 0.1, size=(1, 3))
        pointcloud[:, :3] += translation

        return pointcloud, seg

if __name__ == "__main__":
    dataset = TractorsAndCombines(root='/home/morten/Repos/Pointnet_Pointnet2_pytorch/data/', split='train')
