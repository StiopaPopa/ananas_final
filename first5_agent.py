import torch
import numpy as np
from train_first5_model import First5Model

class First5Agent:
    """Agent that predicts first 5 placements using trained model."""
    def __init__(self, model_path="first5_model.pth"):
        self.model = First5Model()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def act(self, hand):
        """Predicts best placement for given 5-card hand."""
        hand_tensor = torch.tensor(hand, dtype=torch.float).unsqueeze(0)  # Add batch dim
        with torch.no_grad():
            logits = self.model(hand_tensor)  # Shape (1, 5, 3)
        placement = torch.argmax(logits, dim=-1).squeeze(0).tolist()
        return placement  # Returns 5-element list of 0,1,2 actions
