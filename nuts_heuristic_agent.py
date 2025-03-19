'''
Loose re-implementation of Tan & Xiao (2018) initial-nuts-greedy heuristc that
detects and places premium / nutted hands or draws on the bottom, and
distributing the remaining cards by a rank basis (>= 8 : middle, < 8 : top).
Their paper did not consider Fantasyland, so this heuristic only applies for the
first dealing when agent is not in FL. If no such initial hand configuration is
detected (and for all later streets), the agent resorts to the fallback policy.
'''

import numpy as np
from random_agent import RandomAgent
from pineapple_utils import PineappleUtils as P
from collections import defaultdict
from treys import Card

class NutsHeuristicAgent:
    def __init__(self, name="NutsHeuristicAgent", fallback=RandomAgent):
        self.name = name
        # Our fallback policy instance
        self.fallback = fallback()
        # Dictates card -> action based on nuts-greedy heuristic
        self.action = None

    def act(self, observation):
        # 1) If we have a precomputed 'action' array, use it
        if self.action is not None:
            current_card = observation["card"]  # 0..51 or possibly None
            # If there's no current_card (None) or we can't find an assigned row, fallback
            if current_card is None or self.action[current_card] < 0:
                return self.fallback.act(observation)

            # Return the stored row for this card
            chosen_action = self.action[current_card]
            # Mark it used
            self.action[current_card] = -1

            # If we’ve used up all assigned slots, reset
            if np.all(self.action == -1):
                self.action = None

            return chosen_action

        # 2) Otherwise, we might set up the special "first 5 cards" logic
        # We only do this once per deal. We check if agent is NOT in fantasy
        # and exactly 5 cards are in their selection.

        # Gather all cards currently "owned" by the agent in selection
        selection = observation["selection"].copy()
        # Also include the single "observation['card']" in that set
        # (the environment’s logic: 'card' is the one you must place now)
        c = observation["card"]
        if c is not None:
            selection[c] = 1

        # Build a list of Treys int-cards
        cards = [P._int_to_treys(i) for i, have_card in enumerate(selection) if have_card == 1]

        # Check if we’re in the first 5-card deal of a non-fantasy round
        if (not observation["my_fantasy"]) and (len(cards) == 5):
            # We'll create an array to store actions for each of these 5 cards
            self.action = np.full(52, -1, dtype=np.int8)

            # Classify the 5-card combination
            rank, rank_class = P._get_row_rank(cards)

            if rank_class in ["Straight", "Flush", "Full House", "Four of a Kind",
                              "Straight Flush", "Royal Flush"]:
                # Place all cards on the bottom row (action=0)
                for card_treys in cards:
                    card_id = P._treys_to_int(card_treys)
                    self.action[card_id] = 0

            elif rank_class in ["Three of a Kind", "Two Pair"]:
                # Put the pair/trips part in bottom row, others in mid/top
                freq = defaultdict(int)
                for card_treys in cards:
                    freq[Card.get_rank_int(card_treys)] += 1

                for card_treys in cards:
                    rank_int = Card.get_rank_int(card_treys)
                    card_id = P._treys_to_int(card_treys)
                    if freq[rank_int] >= 2:
                        # Part of the pair/trips => bottom
                        self.action[card_id] = 0
                    else:
                        # Single cards
                        if rank_int >= 6:  # (6 + 2 = 8) or something similar
                            self.action[card_id] = 1
                        else:
                            self.action[card_id] = 2

            else:
                # --------------------------------------------------
                # 3) Check for "premium draws" in priority order
                # --------------------------------------------------

                # (a) 4 of same suit => flush draw
                # (b) 4 in a row => open-ended straight draw
                # (c) 3 of same suit => backdoor flush
                # (d) 3 in a row => backdoor straight

                # Suit frequencies
                suit_buckets = defaultdict(list)  # suit_int -> list of treys cards
                for tcard in cards:
                    suit_int = Card.get_suit_int(tcard)  # 1,2,4,8 for c,d,h,s
                    suit_buckets[suit_int].append(tcard)

                # Max suit
                max_suit_count = 0
                max_suit = None
                for s, cardlist in suit_buckets.items():
                    if len(cardlist) > max_suit_count:
                        max_suit_count = len(cardlist)
                        max_suit = s

                # Helper to check consecutive
                # Return the largest consecutive "run" length and which cards are in that run
                consec_length, consec_cards = self._largest_consecutive_run(cards)

                # We'll place "priority" on 4 same suit or 4 consecutive
                # next on 3 same suit or 3 consecutive
                if max_suit_count >= 4:
                    # place those 4 suit cards bottom
                    flush_cards = suit_buckets[max_suit]
                    # If there are exactly 4 of that suit, we place them bottom
                    # leftover card -> mid or top by rank
                    for fc in flush_cards:
                        self.action[P._treys_to_int(fc)] = 0
                    leftover = [x for x in cards if x not in flush_cards]
                    # Place leftover in mid/top
                    for leftover_card in leftover:
                        r = Card.get_rank_int(leftover_card)
                        card_id = P._treys_to_int(leftover_card)
                        if r >= 6:
                            self.action[card_id] = 1
                        else:
                            self.action[card_id] = 2

                elif consec_length >= 4:
                    # place those 4 consecutive in bottom
                    for cc in consec_cards:
                        self.action[P._treys_to_int(cc)] = 0
                    leftover = [x for x in cards if x not in consec_cards]
                    # leftover in mid or top
                    for leftover_card in leftover:
                        r = Card.get_rank_int(leftover_card)
                        card_id = P._treys_to_int(leftover_card)
                        if r >= 6:
                            self.action[card_id] = 1
                        else:
                            self.action[card_id] = 2

                elif max_suit_count == 3:
                    # 3 same suit => backdoor flush
                    flush_cards = suit_buckets[max_suit]
                    for fc in flush_cards:
                        self.action[P._treys_to_int(fc)] = 0
                    leftover = [x for x in cards if x not in flush_cards]
                    for leftover_card in leftover:
                        r = Card.get_rank_int(leftover_card)
                        card_id = P._treys_to_int(leftover_card)
                        if r >= 6:
                            self.action[card_id] = 1
                        else:
                            self.action[card_id] = 2

                elif consec_length == 3:
                    # 3 consecutive => backdoor straight
                    for cc in consec_cards:
                        self.action[P._treys_to_int(cc)] = 0
                    leftover = [x for x in cards if x not in consec_cards]
                    for leftover_card in leftover:
                        r = Card.get_rank_int(leftover_card)
                        card_id = P._treys_to_int(leftover_card)
                        if r >= 6:
                            self.action[card_id] = 1
                        else:
                            self.action[card_id] = 2
                else:
                    # If no recognized pattern, fallback for the entire street
                    self.action = None
                    return self.fallback.act(observation)

            # Now we have assigned each of the 5 cards to a row in 'self.action'.
            # For the *current* call, we just return the row for observation["card"].
            return self.act(observation)

        # 3) If none of the special logic applies, fallback
        return self.fallback.act(observation)
    

    @staticmethod
    def _largest_consecutive_run(cards):
        """
        Utility: find the largest consecutive sequence in 'cards' by rank.
        Return (length, [the subset of cards in that run]).
        
        For simplicity, we ignore "Ace low" (A2345) logic here. 
        If you want that, you'd need extra checks.
        """
        if not cards:
            return 0, []

        # Extract ranks
        ranks = [(Card.get_rank_int(c), c) for c in cards]
        # Sort by rank ascending
        ranks.sort(key=lambda x: x[0])

        # We'll do a sliding approach to find the max consecutive run
        best_len = 1
        best_run = [ranks[0][1]]
        temp_len = 1
        temp_run = [ranks[0][1]]

        for i in range(1, len(ranks)):
            if ranks[i][0] == ranks[i-1][0] + 1:
                temp_len += 1
                temp_run.append(ranks[i][1])
                if temp_len > best_len:
                    best_len = temp_len
                    best_run = temp_run[:]
            elif ranks[i][0] == ranks[i-1][0]:
                # Same rank? We skip duplicates or you can treat them as "break in consecutive"
                continue
            else:
                # reset
                temp_len = 1
                temp_run = [ranks[i][1]]

        return best_len, best_run