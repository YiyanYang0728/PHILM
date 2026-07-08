import yaml
import numpy as np
import pandas as pd
import torch
import sys, os, argparse
from .trainer import train_model
from .optimize import objective
import optuna
import shutil


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def train_model(args):
    # Load configuration
    config = load_config(args.config)

    # Load data
    train_X_data = pd.read_table(config['data']['train_X'], header=None)  # Shape: (samples, bacterial_features)
    train_Y_data = pd.read_table(config['data']['train_Y'], header=None)  # Shape: (samples, viral_features)
    val_X_data = pd.read_table(config['data']['val_X'], header=None)  # Shape: (samples, bacterial_features)
    val_Y_data = pd.read_table(config['data']['val_Y'], header=None)  # Shape: (samples, viral_features)
    
    # Load fixed parameters
    batch_size = config['model']['batch_size']
    patience = config['model']['patience']
    num_epochs = config['model']['num_epochs']
    n_trials = config['model']['n_trials']
    
    # Load target model path and parameter path
    saved_model_path = config['model']['path']
    para_path = config['model']['para_path']
        
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device(0)  # Use GPU 0
    else:
        device = torch.device('cpu') # Fallback to CPU
    print(f'Using device: {device}')

    # Create an Optuna study and optimize
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_X_data, train_Y_data, val_X_data, val_Y_data, batch_size, patience, num_epochs, device), n_trials=n_trials)

    # Print the best parameters, score, and model name
    print("Best Parameters:", study.best_params)
    print("Best Val Loss:", study.best_value)
    print("Best Model Name:", study.best_trial.user_attrs["model_path"])
    
    ori_model_path = study.best_trial.user_attrs["model_path"]
    # os.rename(ori_model_path, saved_model_path)
    os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
    shutil.copy2(ori_model_path, saved_model_path)

    # Save best parameters and model name to a YAML file
    best_params = {
        'learning_rate': study.best_params.get('learning_rate', None),
        'hidden_size': study.best_params.get('hidden_size', None),
        'weight_decay': study.best_params.get('weight_decay', None)
    }
    with open(para_path, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)
    
def main():
    from philm import __version__
    parser = argparse.ArgumentParser(description=f'PHILM {__version__} train model')
    parser.add_argument('-c', '--config', required=True, type=str,
                        help='config file path')
    args = parser.parse_args()    
    train_model(args)

if __name__ == '__main__':
    main()
