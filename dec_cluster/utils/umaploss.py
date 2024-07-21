import torch
import torch.nn as nn


import torch
import torch.nn as nn

class UMAPLoss(nn.Module):
    def __init__(self, spread=1.0, n_neighbors=14):
        super(UMAPLoss, self).__init__()
        self.spread = spread
        self.n_neighbors = n_neighbors

    def compute_adjacency_matrix(self, distances): #计算邻接矩阵，基于距离将矩阵来标记最近的n_neighbors
        _, indices = torch.topk(distances, self.n_neighbors, largest=False, sorted=True)
        adjacency_matrix = torch.zeros_like(distances)
        for i in range(distances.size(0)):
            adjacency_matrix[i, indices[i]] = 1
        return adjacency_matrix

    def forward(self, high_dim_data, low_dim_embedding):
        high_dim_distances = torch.cdist(high_dim_data, high_dim_data, p=2)
        low_dim_distances = torch.cdist(low_dim_embedding, low_dim_embedding, p=2)
        adjacency_matrix = self.compute_adjacency_matrix(high_dim_distances)
        high_dim_similarity = torch.exp(-high_dim_distances / self.spread)
        low_dim_similarity = 1.0 / (1.0 + low_dim_distances)
        loss = -torch.mean(adjacency_matrix * high_dim_similarity * torch.log(low_dim_similarity + 1e-10) 
                           + (1.0 - adjacency_matrix) * (1.0 - high_dim_similarity) * torch.log(1.0 - low_dim_similarity + 1e-10))
        return loss*100
        
        
        
        
        
        
        
        
        
        
        
        
