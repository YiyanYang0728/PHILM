import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# Device configuration
if torch.cuda.is_available():
    device = torch.device(0)  # Use GPU 0
else:
    device = torch.device('cpu')     # Fallback to CPU

def _get_ranks(x):
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp).to(device)
    ranks[tmp] = torch.arange(len(x)).to(device)
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
    corr_list = torch.tensor(corr_list, device=device)
    return corr_list

def bc_sim_list(pred, true):
    numerator = torch.sum(torch.abs(pred - true), dim=1).to(device)
    denominator = torch.sum(pred + true, dim=1).to(device)
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
