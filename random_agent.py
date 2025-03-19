'''
Selects random legal action.
'''

import numpy as np

class RandomAgent:
    def __init__(self, name="RandomAgent"):
        self.name = name

    def act(self, observation):
        """
        observation["action_mask"] is length 4, each entry is 0 or 1.
        Pick a random valid action.
        """
        mask = observation["action_mask"]
        valid_actions = np.flatnonzero(mask)
        action = np.random.choice(valid_actions)
        return action