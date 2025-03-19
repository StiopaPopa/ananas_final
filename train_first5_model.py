import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

DATA_FILE = "expert_first5_data.json"
MODEL_FILE = "first5_model.pth"

class First5Dataset(Dataset):
    """PyTorch Dataset for first 5 card placement learning."""
    def __init__(self, data_file):
        with open(data_file, "r") as f:
            data = json.load(f)
        self.hands = [torch.tensor(d["hand"], dtype=torch.long) for d in data]
        self.placements = [torch.tensor(d["placement"], dtype=torch.long) for d in data]

    def __len__(self):
        return len(self.hands)

    def __getitem__(self, idx):
        return self.hands[idx], self.placements[idx]

class First5Model(nn.Module):
    """Simple feedforward neural net 4 predicting placements."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * 5)  # 3 choices per card, 5 cards => output size 15
        )

    def forward(self, x):
        x = self.fc(x)
        return x.view(-1, 5, 3)  # Reshape to (batch, 5 cards, 3 actions)

def train_model():
    """Train the neural network on expert placements."""
    dataset = First5Dataset(DATA_FILE)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = First5Model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        for hands, placements in dataloader:
            optimizer.zero_grad()
            output = model(hands.float())  # Convert input to float
            loss = criterion(output.view(-1, 3), placements.view(-1))  # CE Loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_model()
