# ppo_agent_bridge.py

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

class NoBustAgent:
    """
    A bridging agent that uses a saved MaskablePPO model
    to pick actions in a multi-agent PettingZoo environment.
    """
    def __init__(self, model_path="no_bust_ppo", name="PPO_NoBust"):
        self.model = MaskablePPO.load(model_path)
        self.name = name

    def act(self, obs):
        """
        pettingzoo_obs is a dict from env.last():
          {
            "card": ...
            "selection": shape(52)
            "top": shape(3)
            "middle": shape(5)
            "bottom": shape(5)
            ...
            "action_mask": shape(4)
            ...
          }
        We'll flatten it just like in single_agent_pineapple, 
        then pass action_masks=... to self.model.predict.
        """
        if obs["action_mask"] is None:
            # or if done/truncated
            return 0  # safe default
        mask = obs["action_mask"]

        # Flatten
        obs_vec = self._flatten_obs(obs)
        # Convert mask from [0..1] to boolean
        bool_mask = (mask == 1)

        # Now call predict
        action, _states = self.model.predict(
            obs_vec, action_masks=bool_mask, deterministic=True
        )
        return int(action)

    def _flatten_obs(self, raw_obs):
        # same logic as single_agent_pineapple:
        obs_list = []

        card_val = raw_obs["card"] if raw_obs["card"] is not None else 52
        obs_list.append(card_val / 52.0)

        obs_list.extend(raw_obs["selection"].astype(np.float32))

        for key in ["top","middle","bottom","opp_top","opp_middle","opp_bottom"]:
            arr = raw_obs[key].astype(np.float32)
            arr = np.clip(arr,0,52)/52.0
            obs_list.extend(arr)

        obs_list.append(float(raw_obs["my_fantasy"]))
        obs_list.append(float(raw_obs["opp_fantasy"]))

        obs_list.extend(raw_obs["unseen"].astype(np.float32))

        return np.array(obs_list, dtype=np.float32)
