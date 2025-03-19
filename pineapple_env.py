import functools
from copy import copy, deepcopy

import gymnasium as gym
import numpy as np

from treys import Card, Deck, Evaluator
from pineapple_utils import PineappleUtils as P

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

class PineappleEnv(AECEnv):
    metadata = {
        "name": "pineapple_env_v0",
        "render_modes": ["human"],
    }

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.possible_agents = ["player_1", "player_2"]

        # Useful shit
        self.button = {"player_1" : True, "player_2" : False}
        self.fantasy = {a : False for a in self.possible_agents}
        self.card = {a : None for a in self.possible_agents}
        self.treys_card = {a : None for a in self.possible_agents}
        self.selection = {a : None for a in self.possible_agents}
        self.treys_selection = {a : None for a in self.possible_agents}
        self.hand = {a : None for a in self.possible_agents}
        self.treys_hand = {a : None for a in self.possible_agents}
        self.row_size = {a : None for a in self.possible_agents}
        self.action_mask = {a : None for a in self.possible_agents}
        self.unseen = {a : None for a in self.possible_agents}
        self.deck = Deck()
        self.evaluator = Evaluator()

        self.rewards = {a : None for a in self.possible_agents}
        self.terminations = {a : None for a in self.possible_agents}
        self.truncations = {a : False for a in self.possible_agents}
        self.infos = {a : None for a in self.possible_agents}

    def other_agent(self, agent):
        for other_agent in self.possible_agents:
            if other_agent != agent:
                return other_agent

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return gym.spaces.Dict({
            # Current card to be acted on
            "card" : gym.spaces.Discrete(52),
            # Binary mask/encoding of whether each card is available or not
            "selection" : gym.spaces.MultiBinary(52),
            # Hand row-by-row. We have 53 since we reserve a # for "empty"
            "top" : gym.spaces.MultiDiscrete([53] * 3),
            "middle" : gym.spaces.MultiDiscrete([53] * 5),
            "bottom" : gym.spaces.MultiDiscrete([53] * 5),
            "opp_top" : gym.spaces.MultiDiscrete([53] * 3),
            "opp_middle" : gym.spaces.MultiDiscrete([53] * 5),
            "opp_bottom" : gym.spaces.MultiDiscrete([53] * 5),
            # Legal actions
            "action_mask" : gym.spaces.MultiBinary(4),
            # Fantasyland
            "my_fantasy" : gym.spaces.MultiBinary(1),
            "opp_fantasy" : gym.spaces.MultiBinary(1),
            # Unseen cards
            "unseen" : gym.spaces.MultiBinary(52)
        })
    
    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called. If both are in FL, cannot see opps.
        """
        other_agent = self.other_agent(agent)
        return {
            "card" : self.card[agent],
            "selection" : self.selection[agent],
            "top" : self.hand[agent][2],
            "middle" : self.hand[agent][1],
            "bottom" : self.hand[agent][0],
            # Only see opp's hand if not both in FL
            "opp_top" : np.full(3, 52, dtype=np.int8) if (self.fantasy[agent] and self.fantasy[other_agent]) else self.hand[other_agent][2],
            "opp_middle" : np.full(5, 52, dtype=np.int8) if (self.fantasy[agent] and self.fantasy[other_agent]) else self.hand[other_agent][1],
            "opp_bottom" : np.full(5, 52, dtype=np.int8) if (self.fantasy[agent] and self.fantasy[other_agent]) else self.hand[other_agent][0],
            # Legal actions
            "action_mask" : self.action_mask[agent],
            # Fantasyland
            "my_fantasy" : self.fantasy[agent],
            "opp_fantasy" : self.fantasy[other_agent],
            # Unseen cards
            "unseen" : self.unseen[agent]
        }

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return gym.spaces.Discrete(4)

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        string = f'''
        ======================
        {self.agent_selection} moves next...
        Player_1:
        Hand:
        {self.treys_hand['player_1'][2]}
        {self.treys_hand['player_1'][1]}
        {self.treys_hand['player_1'][0]}
        Card: {self.treys_card['player_1']}
        Selection: {self.treys_selection['player_1']}
        
        Player_2:
        {self.treys_hand['player_2'][2]}
        {self.treys_hand['player_2'][1]}
        {self.treys_hand['player_2'][0]}
        Card: {self.treys_card['player_2']}
        Selection: {self.treys_selection['player_2']}
        '''

        print(string)

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    '''
    Returns whether specified agent is done with their turn.
    '''
    def is_turn_done(self, agent):
        # Num cards placed in hand
        num_placed = np.sum(self.row_size[agent])
        # If in FL, only done iff placed all 13 cards
        if self.fantasy[agent]:
            return (num_placed == 13)
        # Else done iff placed 5, or [7, 9, 11, 13] cards and have selection
        else:
            return (
                (num_placed == 5 and self.num_actions[agent] == 5)
                or (num_placed in [7, 9, 11, 13] and self.num_actions[agent] == 2)
            )

    '''
    Returns whether round is done for both agents by default, though can specify.
    '''
    def is_done(self, agent):
        # Have placed all of their 13 cards
        return np.sum(self.row_size[agent]) == 13

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.deck.shuffle()

        # For end-of-turn checks
        self.num_actions = {agent : 0 for agent in self.agents}

        # Needed for PettingZoo
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        # Fantasyland
        for agent in self.agents:
            if self.treys_hand[agent] is None:
                break
                
            if not (
                len(self.treys_hand[agent][2]) == 3
                and len(self.treys_hand[agent][1]) == 5
                and len(self.treys_hand[agent][0]) == 5
            ):
                continue

            agent_hand = deepcopy(self.treys_hand[agent])
            for row in range(len(agent_hand)):
                for col in range(len(agent_hand[row])):
                    agent_hand[row][col] = P._str_to_int(agent_hand[row][col])
            self.fantasy[agent] = (
                P._is_fantasyland(agent_hand, self.fantasy[agent])
            )
            # print(agent, self.fantasy[agent])
        
        # Swap button if no FL
        if not (self.fantasy[self.agents[0]] or self.fantasy[self.agents[1]]):
            for agent in self.agents:
                self.button[agent] = not self.button[agent]

        # Get 1st and 2nd acting- & dealt- agents
        first_agent = second_agent = None
        for agent in self.agents:
            if self.button[agent]:
                second_agent = agent
                first_agent = self.other_agent(second_agent)

        # Deal out cards accordingly
        num_first = 14 if self.fantasy[first_agent] else 5
        num_second = 14 if self.fantasy[second_agent] else 5
        self.treys_selection[first_agent] = [
            Card.int_to_str(treys_card) for treys_card in self.deck.draw(num_first)
        ]
        self.treys_selection[second_agent] = [
            Card.int_to_str(treys_card) for treys_card in self.deck.draw(num_second)
        ]
        for agent in self.agents:
            # Unseen (i.e. live) cards
            self.unseen[agent] = np.ones(52, dtype=np.int8)
            for treys_card in self.treys_selection[agent]:
                self.unseen[agent][P._treys_to_int(treys_card)] = 0
            # Get card and selection
            self.treys_card[agent] = self.treys_selection[agent].pop()
            self.card[agent] = P._treys_to_int(self.treys_card[agent])
            self.selection[agent] = np.zeros(52, dtype=np.int8)
            for treys_card in self.treys_selection[agent]:
                self.selection[agent][P._treys_to_int(treys_card)] = 1
        
        # Initialize empty hands and trackers too
        for agent in self.agents:
            self.treys_hand[agent] = [[], [], []]
            # Fill with 52s which indicate "empty slot"
            self.hand[agent] = [np.full(5, 52, dtype=np.int8),
                        np.full(5, 52, dtype=np.int8),
                        np.full(3, 52, dtype=np.int8)]
            self.row_size[agent] = np.zeros(3, dtype=np.int8)
            self.action_mask[agent] = np.ones(4, dtype=np.int8)
            # If not FL, on first deal cannot discard!
            if not self.fantasy[agent]:
                self.action_mask[agent][3] = 0
        
        # Internal returnables
        for agent in self.agents:
            self.rewards[agent] = 0
            self.terminations[agent] = False
            self.infos[agent] = self.observe(agent)
        
        # Select first agent
        self.agent_selection = (
            # It's `first_agent` unless they're in FL
            first_agent if not self.fantasy[first_agent] else second_agent
        )

    def step(self, action):
        # Current acting agent
        agent = self.agent_selection
        
        # 0) Update info
        self.infos[agent] = self.observe(agent)
        
        # 1) Commit action
        if action == 3: 
            # Can no longer discard
            self.action_mask[agent][3] = 0 
        else: # Only act if not discarding!
            self.num_actions[agent] += 1 # We only update this if not discarding
            self.treys_hand[agent][action].append(self.treys_card[agent])
            self.hand[agent][action][self.row_size[agent][action]] = self.card[agent]
            self.row_size[agent][action] += 1
            # Mask action if row full
            if self.row_size[agent][action] == len(self.hand[agent][action]):
                self.action_mask[agent][action] = 0
            else:
                self.action_mask[agent][action] = 1
        
        # 2) Get new selection if needed (i.e. turn done but not round)
        # and unhide ability to discard after every non-FL dealing. We only
        # reach this if we're not in FL and need to be dealt 3 cards.
        if self.is_turn_done(agent) and not self.is_done(agent):
            # Unmask discarding
            self.action_mask[agent][3] = 1 
            # Deal 3 new cards
            self.treys_selection[agent] = [
                Card.int_to_str(treys_card) for treys_card in self.deck.draw(3)
            ]
            # Update unseen for both agents
            other_agent = self.other_agent(agent)
            for treys_card in self.treys_selection[agent]:
                self.unseen[agent][P._treys_to_int(treys_card)] = 0
            for row, row_len in enumerate(self.row_size[agent]):
                for i in range(row_len):
                    self.unseen[other_agent][self.hand[agent][row][i]] = 0
            for row, row_len in enumerate(self.row_size[other_agent]):
                for i in range(row_len):
                    self.unseen[agent][self.hand[other_agent][row][i]] = 0
            
        # 3) Either way we perform these actions:
        # Pop and store card safely
        self.treys_card[agent] = (
            None if not len(self.treys_selection[agent])
            else self.treys_selection[agent].pop()
        )
        self.card[agent] = (
            None if not self.treys_card[agent]
            else P._treys_to_int(self.treys_card[agent])
        )
        # Update selection array
        self.selection[agent] = np.zeros(52, dtype=np.int8)
        for treys_card in self.treys_selection[agent]:
            self.selection[agent][P._treys_to_int(treys_card)] = 1
        
        # 4) Check for termination --> rewards and FL!s
        other_agent = self.other_agent(agent)
        if self.is_done(agent) and self.is_done(other_agent):
            # Only when both are done do we set termination for both to True
            self.terminations[agent] = True
            self.terminations[other_agent] = True
            # Get final hands and convert to Treys ints
            agent_hand, other_agent_hand = deepcopy(self.treys_hand[agent]), deepcopy(self.treys_hand[other_agent])
            for row in range(len(agent_hand)):
                for col in range(len(agent_hand[row])):
                    agent_hand[row][col] = P._str_to_int(agent_hand[row][col])
                    other_agent_hand[row][col] = P._str_to_int(other_agent_hand[row][col])
            # Compute rewards
            self.rewards[agent], self.rewards[other_agent] = P._get_reward(agent_hand, other_agent_hand)
            # Check FL
            self.fantasy[agent] = P._is_fantasyland(agent_hand, self.fantasy[agent])
            self.fantasy[other_agent] = P._is_fantasyland(other_agent_hand, self.fantasy[other_agent])

        # 5) Get next-to-act agent
        self._accumulate_rewards() # Needed for PettingZoo
        if self.is_turn_done(agent):
            # Reset num actions
            self.num_actions[agent] = 0
            # Only swap over if other agent is NOT in FL, or in FL but we're done
            other_agent = self.other_agent(agent)
            if not self.fantasy[other_agent] or self.is_done(agent):
                self.agent_selection = self.other_agent(agent)

        # 6) Render state
        if self.render_mode == "human":
            self.render()
