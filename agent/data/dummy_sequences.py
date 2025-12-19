import torch

def generate_dummy_sequences():
    # (B=4, T=10, F=12)
    X = torch.rand(4, 10, 12)

    # Labels: NORMAL, SUSPICIOUS, DANGER
    y = torch.tensor([0, 1, 2, 0])

    return X, y
