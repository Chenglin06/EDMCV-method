from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, adjusted_rand_score, normalized_mutual_info_score, precision_recall_fscore_support

import torch


def accuracy(model, ds, device):
    truth, pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in ds:
            x = x.to(device)
            x = x.float()
            truth.append(y)
            pred.append(model(x).max(1)[1].cpu())        
            
    
    truth = np.concatenate(truth)
    pred = np.concatenate(pred)

    #confusion_m = confusion_matrix(torch.cat(truth).numpy(), torch.cat(pred).numpy())
    confusion_m = confusion_matrix(truth,pred)

    _, col_idx = linear_sum_assignment(confusion_m, maximize=True)
    acc = np.trace(confusion_m[:, col_idx]) / confusion_m.sum()
    print(f'------------------------acc result----------------------------: {acc}')
    
    f1 = f1_score(truth, pred, average='macro')
    print(f'F1 Score: {f1}')
    
    # 计算Adjusted Rand Index
    ari = adjusted_rand_score(truth, pred)
    print(f'Adjusted Rand Index: {ari}')
    
    # 计算Normalized Mutual Information
    nmi = normalized_mutual_info_score(truth, pred)
    print(f'Normalized Mutual Information: {nmi}')
    
    precision, recall, _, _ = precision_recall_fscore_support(truth, pred, average=None)
    for i, (p, r) in enumerate(zip(precision, recall)):
        print(f'Class {i} - Precision: {p}, Recall: {r}')
        
        
    #return acc
    return acc, f1, ari, nmi, precision, recall
