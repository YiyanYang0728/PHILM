# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader

class MGX_MVX_Dataset(Dataset):
    def __init__(self, train_X_data, train_Y_data):
        self.train_X_data = train_X_data
        self.train_Y_data = train_Y_data

    def __len__(self):
        return len(self.train_X_data)

    def __getitem__(self, idx):
        X = torch.tensor(self.train_X_data.loc[idx], dtype=torch.float32)
        y = torch.tensor(self.train_Y_data.loc[idx], dtype=torch.float32)
        return X, y

class Predict_Dataset(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        X = torch.tensor(self.X_data.loc[idx], dtype=torch.float32)
        return X
