# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from datetime import datetime

from data_loader import MGX_MVX_Dataset
from model import MLP_NODE
import numpy as np
import logging

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, smoothing_factor=0.1, relative_threshold=0.5):
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing_factor = smoothing_factor
        self.relative_threshold = relative_threshold
        self.counter = 0
        self.best_loss = np.inf
        self.smoothed_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        # Smooth the validation loss
        if self.smoothed_loss is None:
            self.smoothed_loss = val_loss
        else:
            self.smoothed_loss = (self.smoothing_factor * val_loss + 
                                  (1 - self.smoothing_factor) * self.smoothed_loss)

        # Check if the smoothed loss has improved
        if self.smoothed_loss < self.best_loss - self.min_delta:
            self.best_loss = self.smoothed_loss
            self.counter = 0
        else:
            # Check if the relative improvement is below the threshold
            relative_change = (self.best_loss - self.smoothed_loss) / self.best_loss
            if relative_change < self.relative_threshold:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.counter = 0

        return self.early_stop
    

def train_model(train_X_data, train_Y_data, val_X_data, val_Y_data, device, 
                num_epochs=200, batch_size=200, learning_rate=0.005,
                hidden_size=512,
                patience = 10,
                weight_decay=1e-6,
                model_path='best_model.pth',
                log_path='train.log'):
    
    # Configure logging
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    # Data
    train_dataset = MGX_MVX_Dataset(train_X_data, train_Y_data)
    val_dataset = MGX_MVX_Dataset(val_X_data, val_Y_data)

    # Model
    model = MLP_NODE(input_dim=train_X_data.shape[1],
                        output_dim=train_Y_data.shape[1],
                        hidden_size=hidden_size).to(device)
    
    logging.info(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    early_stopping = EarlyStopping(patience=patience, min_delta=1e-4, smoothing_factor=0.1, relative_threshold=0.01)

    # use all training data
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # early stopping parameters
    best_val_loss = float('inf')
    counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        tot_samples = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            tot_samples += X_batch.size(0)

        epoch_loss = running_loss / tot_samples

        # Validation
        model.eval()
        val_loss = 0.0
        val_tot_samples = 0
        with torch.no_grad():
            for val_X_batch, val_y_batch in val_loader:
                val_X_batch, val_y_batch = val_X_batch.to(device), val_y_batch.to(device)
                outputs = model(val_X_batch)
                loss = criterion(outputs, val_y_batch)
                val_loss += loss.item() * val_X_batch.size(0)
                val_tot_samples += val_X_batch.size(0)
        val_loss /= val_tot_samples
        
        logging.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}, Val_loss: {val_loss:.6f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            counter = 0
            logging.info(f'New best model saved with val loss: {best_val_loss:.6f}')
        else:
            counter += 1
            if early_stopping(val_loss):
                print(f'Early stopping triggered after {epoch + 1} epochs. The best validation_loss is {best_val_loss}', flush=True)
                logging.info(f'Early stopping triggered after {epoch + 1} epochs. The best validation_loss is {best_val_loss}')
                break

    if counter < patience:
        print(f'Training complete ({num_epochs} epochs). '
                     f'Best val loss: {best_val_loss:.6f}', flush=True)
        logging.info(f'Training complete ({num_epochs} epochs). '
                     f'Best val loss: {best_val_loss:.6f}')
        
    return val_loss
