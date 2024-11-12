from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split


def load_slices(slice_paths, target_size=8):
    slices = [np.array(Image.open(slice_path)) for slice_path in slice_paths]
    num_slices = len(slices)

    if num_slices < target_size:
        # Padding: If the number of slices is less than 8, fill with the last slice
        pad_size = target_size - num_slices
        padding = [slices[-1]] * pad_size
        slices.extend(padding)
        slice_paths.extend([slice_paths[-1]] * pad_size)  # Add and fill the path corresponding to the image
    elif num_slices > target_size:
        # Cropping: If the number of slices is more than 8, only take the middle 8 slices
        start_idx = (num_slices - target_size) // 2
        slices = slices[start_idx:start_idx + target_size]
        slice_paths = slice_paths[start_idx:start_idx + target_size]  # Update path list

    # Stack the slices into a 3D array and adjust its dimensions to (depth, height, width)
    slices = np.stack(slices, axis=0)
    return slices, slice_paths
def load_data(train_list_file):
    data = []
    labels = []
    slice_paths_list = []

    with open(train_list_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            image_folder = parts[0]
            slice_files = parts[1].split(',')
            label = int(parts[2])

            slice_paths = [os.path.join(image_folder, slice_file) for slice_file in slice_files]
            slices, slice_paths = load_slices(slice_paths, target_size=8)
            data.append(slices)
            labels.append(label)
            slice_paths_list.append(slice_paths)

    return data, labels, slice_paths_list

class MedicalDataset(Dataset):
    def __init__(self, data, labels, slice_paths=None, transform=None, split='train'):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.slice_paths = slice_paths
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slices = self.data[idx]
        y = self.labels[idx]

        transformed_slices = []
        for slice in slices:
            slice = self.transform(slice)
            transformed_slices.append(slice)


        x = torch.stack(transformed_slices)
        x = x.permute(1, 0, 2, 3)
        if self.split == 'train':
            return x, torch.tensor(y, dtype=torch.long)
        elif self.split == 'test':
            origin_images = slices.transpose(3, 0, 2, 1)
            return x, torch.tensor(y, dtype=torch.long), origin_images, self.slice_paths[idx]
        elif self.split == 'LBW':
            origin_images = slices.transpose(3, 0, 2, 1)
            return x, origin_images, self.slice_paths[idx]

def load_trian_data(train_list_file, batch_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data, labels, slice_paths = load_data(train_list_file)
    dataset = MedicalDataset(data, labels, transform=transform, split='train')
    dataloader = DataLoader(dataset, batch_size, shuffle=False)
    return dataloader

def load_trian_val_data(train_list_file, batch_size, random_state=44):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    datas, labels, slice_paths = load_data(train_list_file)
    data, val_data, labels, val_labels = train_test_split(datas, labels, test_size=0.1, random_state=random_state)
    # Create datasets and data loaders
    train_dataset = MedicalDataset(data, labels, transform=transform, split='train')
    val_dataset = MedicalDataset(val_data, val_labels, transform=transform, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)
    return train_dataloader, val_dataloader
def load_test_data(test_list_file, batch_size=1):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data, labels, slice_paths = load_data(test_list_file)

    dataset = MedicalDataset(data, labels, slice_paths, transform, split='test')
    dataloader = DataLoader(dataset, batch_size, shuffle=False)
    return dataloader


class MedicalDataset_LBW(Dataset):
    def __init__(self, data, slice_paths=None, transform=None):
        self.data = data

        self.transform = transform
        self.slice_paths = slice_paths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slices = self.data[idx]

        transformed_slices = []
        for slice in slices:
            # If your slice is already multi-channel, apply the transformation directly
            slice = self.transform(slice)
            transformed_slices.append(slice)

        # Stacking slices back into 3D tensors (D, C, H, W)
        x = torch.stack(transformed_slices)
        x = x.permute(1, 0, 2, 3)  # Rearrange dimensions to (C, D, H, W)
        origin_images = slices.transpose(3, 0, 2, 1)
        return x, origin_images, self.slice_paths[idx]

def read_LBWTXT_data(train_list_file):
    data = []
    slice_paths_list = []

    with open(train_list_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            image_folder = parts[0]
            slice_files = parts[1].split(',')


            slice_paths = [os.path.join(image_folder, slice_file) for slice_file in slice_files]
            slices, slice_paths = load_slices(slice_paths, target_size=8)
            data.append(slices)
            slice_paths_list.append(slice_paths)

    return data, slice_paths_list  # Keep as a list

def load_LBW_data(train_list_file, batch_size=1):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standardization
    ])

    data, slice_paths = read_LBWTXT_data(train_list_file)

    # Create datasets and data loaders
    dataset = MedicalDataset_LBW(data, slice_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=False)
    return dataloader

if __name__ == '__main__':
    dataloader = load_LBW_data(r'dataset/test_lbw/processe/data_list.txt')
    for batch_idx, (data, origin_images, slice_paths) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"Data shape: {data.shape}")
        print(f"Slice paths: {slice_paths}")
        print()

