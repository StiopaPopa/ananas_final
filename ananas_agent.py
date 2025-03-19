import numpy as np
from ppo_agent_bridge import PPOAgent
from first5_agent import First5Agent
from pineapple_utils import PineappleUtils as P
from collections import defaultdict
from treys import Card

# AnanasAgent now checks for a last-street condition.
# We trigger last-street search if we are not in fantasyland AND if the total free slots (marked by 52)
# across bottom, middle, and top is exactly 2.
class AnanasAgent:
    def __init__(self, name="AnanasAgent"):
        self.name = name
        # Flag for whether we are in the first 5 cards phase.
        self.first_5 = True
        # Stores card -> action mapping produced by First5Agent.
        self.action = None

    def act(self, observation):
        # If in fantasyland, immediately switch to PPO-based actions.
        if observation["my_fantasy"]:
            self.first_5 = False

        # First-5 imitation learning phase.
        if self.first_5:
            current_card = observation["card"]
            selection_mask = observation["selection"]
            # Convert selection mask (52-dimensional binary vector) to list of selected card indices.
            selected_cards = np.where(selection_mask == 1)[0].tolist()
            selected_cards.append(current_card)  # Include current card to form the 5-card set.
            selected_cards.sort()

            # Generate placements using First5Agent on the first call.
            if self.action is None:
                self.action = {}
                output = First5Agent().act(selected_cards)
                # Map each card (by its integer representation) to a placement action.
                for i, card in enumerate(selected_cards):
                    self.action[card] = output[i]

            # Pop the placement for the current card.
            res = self.action.pop(current_card)
            # When all first-5 placements are done, transition to PPO.
            if len(self.action) == 0:
                self.first_5 = False
            return res

        # Determine the number of free slots (empty slots marked by 52) in our hand.
        free_slots = (
            np.sum(np.array(observation["bottom"]) == 52) +
            np.sum(np.array(observation["middle"]) == 52) +
            np.sum(np.array(observation["top"]) == 52)
        )

        # If we're not in fantasyland and exactly 2 free slots remain, we're on the last street.
        if not observation["my_fantasy"] and free_slots == 2 and np.sum(observation["selection"]) == 2:
            print("Triggering last street search...")
            from last_street_search import LastStreetSearch
            searcher = LastStreetSearch()
            # Build our agent state and the opponent's state from the observation.
            agent_state = self.build_agent_state(observation)
            opp_state = self.build_opponent_state(observation)
            # Since we are the last mover, we set am_last_move=True.
            best_candidate, best_value = searcher.best_move(agent_state, opp_state, am_last_move=True)
            # Determine the index of the current card within our candidate selection.
            current_card = observation["card"]
            selection = np.where(observation["selection"] == 1)[0].tolist()
            selection.append(current_card)
            selection.sort()
            current_index = selection.index(current_card)
            # If the candidate move says to discard this card, return action 3.
            # Otherwise, return the row placement from the candidate's mapping.
            if best_candidate["discard"] == current_index:
                action = 3  # discard action.
            else:
                action = best_candidate["placements"][current_index]
            return action

        # For all other situations, defer to the PPO agent.
        return PPOAgent().act(observation)

    def build_agent_state(self, observation):
        """
        Construct a state dictionary for our agent for search simulation.
        Expected keys:
          - 'selection': list of candidate card ints (from selection mask plus current card)
          - 'hand': list of three lists representing bottom, middle, and top rows
          - 'row_size': list of three ints representing how many cards are already placed in each row.
        """
        # Get candidate cards from the selection mask and current card.
        selection = np.where(observation["selection"] == 1)[0].tolist()
        current_card = observation["card"]
        if current_card is not None:
            selection.append(current_card)
        selection.sort()

        # Build the hand from observation keys.
        hand = [
            observation["bottom"].tolist(),
            observation["middle"].tolist(),
            observation["top"].tolist()
        ]
        # Compute row sizes (number of non-empty slots, assuming empty is marked by 52).
        row_size = [
            int(np.sum(np.array(observation["bottom"]) != 52)),
            int(np.sum(np.array(observation["middle"]) != 52)),
            int(np.sum(np.array(observation["top"]) != 52))
        ]
        return {"selection": selection, "hand": hand, "row_size": row_size}

    def build_opponent_state(self, observation):
        """
        Construct a state dictionary for the opponent for search simulation.
        Expected keys similar to our agent's state.
        """
        hand = [
            observation["opp_bottom"].tolist(),
            observation["opp_middle"].tolist(),
            observation["opp_top"].tolist()
        ]
        row_size = [
            int(np.sum(np.array(observation["opp_bottom"]) != 52)),
            int(np.sum(np.array(observation["opp_middle"]) != 52)),
            int(np.sum(np.array(observation["opp_top"]) != 52))
        ]
        # If opponent's selection and current card are not provided, set defaults.
        selection = []  # This could be updated if known.
        card = 52       # Default "empty" card.
        unseen = observation.get("unseen", np.ones(52, dtype=np.int8)).tolist()
        fantasy = observation.get("opp_fantasy", False)
        return {"hand": hand, "row_size": row_size, "selection": selection, "card": card, "fantasy": fantasy, "unseen": unseen}
