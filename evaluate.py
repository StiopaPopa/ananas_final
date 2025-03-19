import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from pineapple_env import PineappleEnv
from pineapple_utils import PineappleUtils as P 

def evaluate_agents(agent1, agent2, num_episodes=10_000, seed=None, mode=None):
    """
    Runs a series of episodes where agent1 is the first in env.agents[0] 
    and agent2 is env.agents[1], then collects performance metrics.
    Returns a dict of stats for each agent.
    """
    # Accumulators
    total_points = {agent1.name: 0.0, agent2.name: 0.0}
    total_royalty = {agent1.name: 0.0, agent2.name: 0.0}
    foul_count   = {agent1.name: 0,   agent2.name: 0  }
    fantasy_count= {agent1.name: 0,   agent2.name: 0  }

    env = PineappleEnv(render_mode=mode)

    flag = False

    for ep in tqdm(range(num_episodes)):
        if flag:
            break
        # print("Episode: ", ep)
        env.reset()

        # Use agent_iter to step through each agent
        for active_agent in env.agent_iter():
            obs, rew, term, trunc, info = env.last()
            if term or trunc:
                break
            else:
                if active_agent == env.agents[0]:
                    action = agent1.act(obs)
                else:
                    action = agent2.act(obs)
                env.step(action)

        # After all agents are done, gather final royalties & stats
        for i, agent_id in enumerate(env.agents):
            # Which agent object is this?
            current_agent_obj = agent1 if i == 0 else agent2
            current_agent_name = current_agent_obj.name

            # Add up final reward
            total_points[current_agent_name] += env.rewards[agent_id]

            # Check if final hand is fouled or in fantasy
            # environment has something like env.treys_hand[agent_id] at the end
            final_hand_str = env.treys_hand[agent_id]  # list of lists of card strings
            if final_hand_str is not None:
                # Convert each string to int for your utility’s usage
                final_hand_int = []
                for row_cards in final_hand_str:
                    row_as_ints = [P._str_to_int(c) for c in row_cards]
                    final_hand_int.append(row_as_ints)

                if P._is_foul(final_hand_int):
                    foul_count[current_agent_name] += 1
                else:
                    # Add final royalty if no foul (not incl. deduction from opp)
                    total_royalty[current_agent_name] += P._total_royalty(final_hand_int)
    
                # Check if ended up in FL:
                if P._is_fantasyland(final_hand_int, env.fantasy[agent_id]):
                    fantasy_count[current_agent_name] += 1


    # Compute final statistics
    results = {}
    for agent_obj in [agent1, agent2]:
        name = agent_obj.name
        results[name] = {
            "avg_points (points)"    : total_points[name] / num_episodes,
            "avg_royalty (points)"    : total_royalty[name] / num_episodes,
            "foul_rate (%)"     : 100.0 * foul_count[name] / num_episodes,
            "fantasy_rate (%)"  : 100.0 * fantasy_count[name] / num_episodes
        }
    return results


if __name__ == "__main__":
    # Import agents
    import torch
    from dqn_agent import DQNAgent, QNetwork
    from random_agent import RandomAgent
    from nuts_heuristic_agent import NutsHeuristicAgent
    from ppo_agent_bridge import PPOAgent
    from no_bust_agent_bridge import NoBustAgent
    from ananas_agent import AnanasAgent

    # DQN prep
    obs_dim = 133
    act_dim = 4
    q_net = QNetwork(obs_dim, act_dim)
    q_net.load_state_dict(torch.load("dqn_pineapple.pt", map_location="cpu"))
    q_net.eval()


    # Instantiate two agents

    # agent1 = RandomAgent(name="RandomA")
    # agent1 = DQNAgent(q_network=q_net, device="cpu")
    # agent1.name = "DQNa"
    # agent2 = DQNAgent(q_network=q_net, device="cpu")
    # agent2.name = "DQNb"
    # agent1 = PPOBridgeAgent("maskable_ppo_pineapple.zip", name="PPO_Agent_a")
    # agent1 = RandomAgent()
    # agent1 = PPOAgent()
    # agent2 = NutsHeuristicAgent(name="Nuts", fallback=PPOAgent)
    # agent2 = PPOAgent(name="B")
    # agent2 = NoBustAgent()
    # agent1 = PPOAgent(name="PPO")
    agent1 = RandomAgent(name="Random")
    agent2 = AnanasAgent(name="Ananas")
    # agent2 = AnanasAgent()
    # agent2 = RandomAgent(name="RandomB")
    # agent1 = NutsHeuristicAgent(name="NutsHeuristicA", fallback=RandomAgent)
    # agent2 = NutsHeuristicAgent(name="NutsHeuristic", fallback=RandomAgent)

    # Evaluate them
    mode = "human" if input("Render? (y/n) ") == "y" else None
    stats = evaluate_agents(agent1, agent2, mode=mode)

    # Print results
    df = pd.DataFrame(stats).T  # transpose so each agent is a row
    print("\nEvaluation Results:")
    print(df)
