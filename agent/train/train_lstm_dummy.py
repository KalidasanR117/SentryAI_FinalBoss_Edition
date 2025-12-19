import torch
import torch.nn as nn

from agent.models.decision_lstm import TemporalDecisionAgent
from agent.data.synthetic_sequences import generate_dataset

# Load synthetic dataset
X, y = generate_dataset()
# X: (B, T, F)
# y: (B,)

print("Dataset shape:", X.shape, y.shape)

model = TemporalDecisionAgent(input_dim=X.shape[2])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(300):
    optimizer.zero_grad()

    preds = model(X)
    loss = criterion(preds, y)

    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

print("\nTraining complete.")
