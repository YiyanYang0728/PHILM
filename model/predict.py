import torch
import sys, yaml, argparse
import pandas as pd
from torch.utils.data import DataLoader

from data_loader import Predict_Dataset
from model import MLP_NODE
from utils import choose_device, empty_device_cache

device = choose_device()
    
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def main(args):
    # Load configuration
    config = load_config(args.config)
    global device
    device = choose_device(config.get('runtime', {}).get('device', 'auto'))
    print(f'Using device: {device}')

    # read data
    test_X_data = pd.read_table(config['data']['test_X'], header=None)
    dataset = Predict_Dataset(test_X_data)

    # load model
    # define hyperparameters (must be the same as during training)
    input_dim = test_X_data.shape[1]  # Should match training input_dim
    output_dim = config['model']['output_dim'] # Use the output_dim from training
    batch_size=config['model']['batch_size']
    hidden_size=config['model']['hidden_size']

    # load data
    test_loader = DataLoader(dataset, batch_size=batch_size)

    model = MLP_NODE(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_size=hidden_size
    )
    model.load_state_dict(torch.load(config['model']['path'], weights_only=False, map_location=device))
    model.to(device)
    model.eval()

    preds = torch.empty((0, output_dim), dtype=torch.float32, device=device)
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            
            pred = model(X_batch)
            pred = pred.to(device)
            preds = torch.cat((preds, pred), 0)
            empty_device_cache(device)
            
    prediction = preds

    print(prediction.shape)
    prefix=config['output_prefix']
    dt = pd.DataFrame(prediction.cpu().numpy())
    dt.to_csv(prefix+"_pred.tsv", sep="\t", header=False, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PHILM')
    parser.add_argument('-c', '--config', required=True, type=str,
                        help='config file path')
    args = parser.parse_args()
    main(args)
