import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# utils.py
# Place any utility functions or helper classes here if needed.

# Example utility function (not used in the current setup):
def example_utility_function():
    pass
