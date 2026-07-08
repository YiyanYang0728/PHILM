# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader

class MGX_MVX_Dataset(Dataset):
    def __init__(self, train_X_data, train_Y_data):
        self.train_X_data = train_X_data.to_numpy(dtype="float32", copy=True)
        self.train_Y_data = train_Y_data.to_numpy(dtype="float32", copy=True)

    def __len__(self):
        return len(self.train_X_data)

    def __getitem__(self, idx):
        X = torch.from_numpy(self.train_X_data[idx])
        y = torch.from_numpy(self.train_Y_data[idx])
        return X, y

class Predict_Dataset(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data.to_numpy(dtype="float32", copy=True)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        X = torch.from_numpy(self.X_data[idx])
        return X
