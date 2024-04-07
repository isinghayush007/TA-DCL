import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from dataloaders.data_utils import get_unk_mask_indices, image_loader


class ChestmnistDataset(Dataset):

    def __init__(self, split, data_file, transform=None):

        self.split = split
        self.split_data = np.load(data_file, allow_pickle=True)
        self.root = "/kaggle/input/oia-odir-5k/oia-odir"

        if self.split == 'train':
            self.imgs = self.split_data['train_images_sick']
            self.labs = self.split_data['train_labels_sick']
        elif self.split == 'val':
            self.imgs = self.split_data['val_images']
            self.labs = self.split_data['val_labels']
        elif self.split == 'test':
            self.imgs = self.split_data['test_images']
            self.labs = self.split_data['test_labels']
        else:
            raise ValueError

        self.transform = transform

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.root + "/" + self.imgs[idx])
        # image = torch.Tensor(np.array(image))
        # if len(image.shape) > 2:
        #     # image = image[:, :, 0]
        #     image = image.unsqueeze(0)
        # if self.transform is not None:
        #     image = self.transform(image)


        if len(image.getbands()) > 1:
            image = image.convert("RGB")  # Convert to RGB if the image has multiple channels

        image = image.resize((640, 640))
    
        image = torch.Tensor(np.array(image))
        
        if self.transform is not None:
            image = self.transform(image)

        if len(image.shape) > 2:
            image = image[:, :, 0]

        labels = self.labs[idx].astype(int)
        labels = torch.Tensor(labels)

        sample = {}
        sample['image'] = image
        sample['labels'] = labels

        return sample


