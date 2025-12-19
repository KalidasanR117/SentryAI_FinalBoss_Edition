# sentry/core/action_model.py
import torch
import torch.nn as nn

class PoseLSTM(nn.Module):
    def __init__(self, num_joints=17, num_classes=3, hidden_size=128):
        super().__init__()

        self.input_size = num_joints * 2  # (x,y) per joint
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x: (batch, frames, joints, 2)
        """
        B, T, J, C = x.shape
        x = x.view(B, T, J * C)   # (batch, frames, 34)

        _, (hn, _) = self.lstm(x)
        out = hn[-1]              # last layer hidden state
        return self.fc(out)
