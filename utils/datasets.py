import pandas as pd
import tifffile as tiff
import torch
from torch.utils.data import Dataset


class SimBADataset(Dataset):
    def __init__(self, csv_file_path, img_dir, no_bias, transform=None):
        self.csv_file_path = csv_file_path
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file_path, low_memory=True)
        self.transform = transform

        if no_bias:
            self.df['bias_label'].values[:] = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['filename'].replace('nii.gz', 'tiff')
        img = tiff.imread(self.img_dir / img_name).astype('f4')

        if self.transform:
            img = self.transform(img)

        return self.df.iloc[idx]['class_label'], self.df.iloc[idx]['bias_label'], torch.unsqueeze(img, 0), \
            self.df.iloc[idx]['filename']
