# train_maskable_ppo.py

import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks

from single_agent_pineapple import SingleAgentPineapple
from random_agent import RandomAgent
from ppo_agent_bridge import PPOAgent


def mask_fn(env: SingleAgentPineapple) -> np.ndarray:
    """
    Called each time an action is sampled or learned by ActionMasker.
    Must return shape (4,) bool indicating valid actions.
    """
    return env.valid_action_mask()

def main():
    # 1) Create single-agent environment: hero is "player_1", opponent random
    env = SingleAgentPineapple(hero_id="player_1", opponent=PPOAgent)

    # 2) Wrap with ActionMasker
    env = ActionMasker(env, mask_fn)

    # 3) Create MaskablePPO model
    model = MaskablePPO("MlpPolicy", env, verbose=1)

    # 4) Train
    model.learn(total_timesteps=5_000_000)

    # 5) Evaluate/test with a short loop
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        # fetch action mask if you want to pass manually
        action_masks = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = (terminated or truncated)

    print("Test episode reward:", total_reward)

    # 6) Save
    model.save("no_bust_ppo")
    print("Model saved to no_bust_ppo.zip")

if __name__ == "__main__":
    main()
