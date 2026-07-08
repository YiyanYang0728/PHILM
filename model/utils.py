import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

def choose_device(device_arg='auto'):
    if device_arg != 'auto':
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def empty_device_cache(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps' and hasattr(torch, 'mps'):
        torch.mps.empty_cache()


def _get_ranks(x):
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x), device=x.device)
    return ranks

def spearman_corr(x, y):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    
    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)

def spearman_corr_list(pred, true):
    # check sample SCC
    corr_list = []
    with torch.no_grad():
        for i, x in enumerate(pred):
            y = true[i]
            corr_i = spearman_corr(x, y)
            corr_list.append(corr_i)
    return torch.stack(corr_list)

def bc_sim_list(pred, true):
    numerator = torch.sum(torch.abs(pred - true), dim=1)
    denominator = torch.sum(pred + true, dim=1)
    bc_sim = 1 - numerator / denominator
    return bc_sim

def rsquared_list(pred, true):
    from sklearn.metrics import r2_score
    pred = pred.cpu().numpy()
    true = true.cpu().numpy()
    n_samples = pred.shape[0]
    r2_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        r2_scores[i] = r2_score(true[i], pred[i])
    return r2_scores

def cos_sim_list(pred, true):
    cos_sim_lst  = nn.CosineSimilarity(dim=1)(pred, true)
    return cos_sim_lst

def pearson_corr_list(pred, true):
    pred = pred.cpu().numpy()
    true = true.cpu().numpy()
    pcc_list = []
    for i in range(pred.shape[0]):
        corr, _ = pearsonr(pred[i,:], true[i,:])
        pcc_list.append(corr)
    return pcc_list

def r2score_list(pred, true):
    pred = pred.cpu().numpy()
    true = true.cpu().numpy()
    r2_list = []
    for i in range(pred.shape[0]):
        model = LinearRegression()
        x=pred[i, :].reshape(-1, 1)
        y=true[i, :].reshape(-1, 1)
        model.fit(x, y)
        r2 = model.score(x, y)
        r2_list.append(r2)
    return r2_list

def ft_cos_sim_list(pred, true):
    cos_sim_lst  = nn.CosineSimilarity(dim=0)(pred, true)
    return cos_sim_lst

def ft_pearson_corr_list(pred, true):
    pred = pred.cpu().numpy()
    true = true.cpu().numpy()
    pcc_list = []
    for i in range(pred.shape[1]):
        corr, _ = pearsonr(pred[:,i], true[:,i])
        pcc_list.append(corr)
    return pcc_list

def ft_r2score_list(pred, true):
    pred = pred.cpu().numpy()
    true = true.cpu().numpy()
    r2_list = []
    for i in range(pred.shape[1]):
        model = LinearRegression()
        x=pred[:, i].reshape(-1, 1)
        y=true[:, i].reshape(-1, 1)
        model.fit(x, y)
        r2 = model.score(x, y)
        r2_list.append(r2)
    return r2_list
