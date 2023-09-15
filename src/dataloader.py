import os
import xml.etree.ElementTree as ET  # For .osm XML files
import json  # For JSON files
import torch
import pandas as pd
from torch.utils.data import Dataset

class OSMDataset(Dataset):
    def __init__(self, dir_path):
        """Initialize dataset

        Args:
            - csv_path (str): Path to the CSV file containing paths to the OSM data files.
        """
        self.dir_path = dir_path
        self.file_list = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                          if f.endswith('.osm') or f.endswith('.json')]

    def _load_single_file(self, idx):
        """
        Load a single OSM data file given its index in the file list.
        
        Parameters:
            - idx (int): Index of the file to load from the list.
        
        Return:
            - Data content of the file.
        """
        data_path = self.file_list[idx]

        if data_path.endswith('.osm'):
            tree = ET.parse(data_path)
            root = tree.getroot()
            return root

        elif data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
            return data

        else:
            raise ValueError("Unsupported file format")

    def __len__(self):
        """
        Get the total number of data files.
        
        Return:
            - Total number of data files.
        """
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Get a single data item from the dataset.
        
        Parameters:
            - idx (int): Index of the data file to load.
        
        Return:
            - Data from the file at the given index.
        """
        raw_data = self._load_single_file(idx)
        return raw_data
        

# osm_dataset = OSMDataset("./data/")
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from torch import nn, optim

# Your Satellite dataset class
class SatelliteDataset(Dataset):
    def __init__(self, satellite_image_paths):
        self.satellite_image_paths = satellite_image_paths

    def __len__(self):
        return len(self.satellite_image_paths)

    def __getitem__(self, index):
        with rasterio.open(self.satellite_image_paths[index]) as src:
            satellite_image = src.read().astype('float32')
        return torch.tensor(satellite_image)

# Your GroundCoverClass dataset class
class GccDataset(Dataset):
    def __init__(self, ground_cover_paths):
        self.ground_cover_paths = ground_cover_paths

    def __len__(self):
        return len(self.ground_cover_paths)

    def __getitem__(self, index):
        with rasterio.open(self.ground_cover_paths[index]) as src:
            ground_cover_image = src.read().astype('float32')
        return torch.tensor(ground_cover_image)

# Your super dataset class that collates both
class SuperDataset(Dataset):
    def __init__(self, satellite_dataset, gcc_dataset):
        self.satellite_dataset = satellite_dataset
        self.gcc_dataset = gcc_dataset

    def __len__(self):
        return len(self.satellite_dataset)

    def __getitem__(self, index):
        satellite_image = self.satellite_dataset[index]
        ground_cover_image = self.gcc_dataset[index]
        return satellite_image, ground_cover_image


