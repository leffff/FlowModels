import torch


def l2_dist(a: torch.Tensor, b: torch.Tensor):
    flat_a = a.view(a.size(0), -1)
    flat_b = b.view(b.size(0), -1)
    
    # Calculate the squared differences
    squared_diff = (flat_a.unsqueeze(1) - flat_b.unsqueeze(0)).pow(2)
    
    # Sum along the channel dimension
    sum_squared_diff = squared_diff.sum(dim=2)
    
    # Take the square root to get Euclidean distances
    distance_matrix = sum_squared_diff.sqrt()
    
    return distance_matrix
