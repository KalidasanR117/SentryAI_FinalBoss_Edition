import torch

def generate_sequence(label, T=10, F=12):
    seq = torch.zeros(T, F)

    for t in range(T):
        if label == 0:  # NORMAL
            seq[t] = torch.tensor([
                0, 1, 0.4,   # object
                0.1, 0.1, 0.2,  # pose
                1, 0.3, 2.5,    # tracking
                1.0, 0,         # identity
                1.0             # duration
            ])
        elif label == 1:  # SUSPICIOUS
            seq[t] = torch.tensor([
                1, 1, 0.6,
                0.4 + 0.05*t, 0.5, 0.6,
                1, 0.6, 1.8,
                0.0, 0,
                2.0
            ])
        else:  # DANGER
            seq[t] = torch.tensor([
                1, 2, 0.9,
                0.6 + 0.03*t, 0.7 + 0.02*t, 0.8,
                2, 1.2, 0.6,
                0.0, 1,
                4.0
            ])

    return seq


def generate_dataset():
    X, y = [], []
    for label in [0, 1, 2]:
        for _ in range(20):
            X.append(generate_sequence(label))
            y.append(label)

    return torch.stack(X), torch.tensor(y)
