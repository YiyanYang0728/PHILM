import torch
import sys, yaml, argparse
import pandas as pd
from torch.utils.data import DataLoader

from data_loader import MGX_MVX_Dataset
from model import MLP_NODE
from utils import choose_device, empty_device_cache, ft_pearson_corr_list, ft_cos_sim_list, ft_r2score_list, pearson_corr_list, cos_sim_list, r2score_list

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
    test_Y_data = pd.read_table(config['data']['test_Y'], header=None)
    dataset = MGX_MVX_Dataset(test_X_data, test_Y_data)

    # load model
    # define hyperparameters (must be the same as during training)
    input_dim = test_X_data.shape[1]  # Should match training input_dim
    output_dim = test_Y_data.shape[1]  # Use the output_dim from training
    
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
    grounds = torch.empty((0, output_dim), dtype=torch.float32, device=device)
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            pred = model(X_batch)
            pred = pred.to(device)
            preds = torch.cat((preds, pred), 0)
            grounds = torch.cat((grounds, Y_batch), 0)
            empty_device_cache(device)
            
    prediction = preds
    ground_truth = grounds

    print(prediction.shape)
    prefix=config['output_prefix']
    dt = pd.DataFrame(prediction.cpu().numpy())
    dt.to_csv(prefix+"_pred.tsv", sep="\t", header=False, index=False)

    # feature-wise metrics
    pcc_lst = ft_pearson_corr_list(prediction, ground_truth)
    pcc_dt = pd.DataFrame(pcc_lst)
    pcc_dt.to_csv(prefix+"_pcc_ft.tsv", sep="\t", header=["PCC"], index=False)

    cos_lst = ft_cos_sim_list(prediction, ground_truth)
    cos_dt = pd.DataFrame(cos_lst.cpu().numpy())
    cos_dt.to_csv(prefix+"_cos_sim_ft.tsv", sep="\t", header=["Cos_Sim"], index=False)

    r2_lst = ft_r2score_list(prediction, ground_truth)
    r2_dt = pd.DataFrame(r2_lst)
    r2_dt.to_csv(prefix+"_r2_ft.tsv", sep="\t", header=["R2"], index=False)

    # sample-wise metrics
    pcc_lst = pearson_corr_list(prediction, ground_truth)
    pcc_dt = pd.DataFrame(pcc_lst)
    pcc_dt.to_csv(prefix+"_pcc.tsv", sep="\t", header=["PCC"], index=False)

    cos_lst = cos_sim_list(prediction, ground_truth)
    cos_dt = pd.DataFrame(cos_lst.cpu().numpy())
    cos_dt.to_csv(prefix+"_cos_sim.tsv", sep="\t", header=["Cos_Sim"], index=False)

    r2_lst = r2score_list(prediction, ground_truth)
    r2_dt = pd.DataFrame(r2_lst)
    r2_dt.to_csv(prefix+"_r2.tsv", sep="\t", header=["R2"], index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PHILM')
    parser.add_argument('-c', '--config', required=True, type=str,
                        help='config file path')
    args = parser.parse_args()
    main(args)
