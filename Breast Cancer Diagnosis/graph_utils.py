import torch
from torch_geometric.data import Data

# Function to Create Graphs
def create_graph(features):
    num_nodes = features.size(0)  # Number of nodes
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()  # Create edges
    return Data(x=features, edge_index=edge_index)
