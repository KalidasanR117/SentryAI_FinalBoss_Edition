import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalDecisionAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        logits = self.fc(h_n[-1])
        return logits   # NO softmax here

