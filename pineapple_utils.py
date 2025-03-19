'''
Contains several Pineapple OFC-relevant helper functions like checking for FL
or comparing rows and hands.
'''

from treys import Deck, Card, Evaluator
import numpy as np
from copy import deepcopy

evaluator = Evaluator()
# Rewards
bottom_royalty_map = {
    "Straight": 2,
    "Flush": 4,
    "Full House": 6,
    "Four of a Kind": 10,
    "Straight Flush": 15,
    "Royal Flush": 25
}
middle_royalty_map = {
    "Three of a Kind": 2,
    "Straight": 4,
    "Flush": 8,
    "Full House": 12,
    "Four of a Kind": 20,
    "Straight Flush": 30,
    "Royal Flush": 50
}
top_royalty_map = {
    "66": 1, 
    "77": 2, 
    "88": 3, 
    "99": 4, 
    "TT": 5, 
    "JJ": 6, 
    "QQ": 7, 
    "KK": 8, 
    "AA": 9,
    "222": 10, 
    "333": 11, 
    "444": 12, 
    "555": 13, 
    "666": 14, 
    "777": 15, 
    "888": 16, 
    "999": 17, 
    "TTT": 18, 
    "JJJ": 19, 
    "QQQ": 20, 
    "KKK": 21, 
    "AAA": 22
}

class PineappleUtils():
    def __init__(self):
        pass
    
    @classmethod
    def _treys_to_int(self, treys_card):
        # If string convert to int
        if isinstance(treys_card, str):
            treys_card = Card.new(treys_card)
        rank_int = Card.get_rank_int(treys_card) 
        suit_int = int(np.log2(Card.get_suit_int(treys_card))) # Treys does powers of 2
        return rank_int * 4 + suit_int
    
    @classmethod
    def _int_to_treys(self, card_i):
        """
        Reverses _treys_to_int. 
        Given a 0..51 integer, return the Treys internal integer representation.
        """
        # 1) Extract rank_index (0..12) and suit_index (0..3)
        rank_index = card_i // 4        # 0..12
        suit_index = card_i % 4         # 0..3

        # 2) Convert to rank (2..14) and suit char (c/d/h/s)
        rank_int = rank_index + 2       # 2..14

        # Map from rank integer to Treys single-character
        rank_map = {
            2: '2', 3: '3', 4: '4', 5: '5',  6: '6',
            7: '7', 8: '8', 9: '9', 10: 'T', 11: 'J',
            12: 'Q', 13: 'K', 14: 'A'
        }

        # Map from suit_index -> suit character
        suit_map = {
            0: 's', 
            1: 'h',   
            2: 'd',  
            3: 'c'   
        }

        rank_char = rank_map[rank_int]
        suit_char = suit_map[suit_index]

        # 3) Build the 2-character string for Treys and convert to Treys int
        card_str = rank_char + suit_char
        treys_card = Card.new(card_str)

        return treys_card
    
    @classmethod
    def _treys_to_ints(self, treys_cards):
        return [self._treys_to_int(treys_card) for treys_card in treys_cards]
    
    @classmethod
    def _str_to_int(self, str):
        return Card.new(str)

    @classmethod
    def _extend_top(self, top):
        suits = ['h', 'd', 's', 'c']
        card_val = 2
        while len(top) < 5:
            for suit in suits:
                card = Card.new(f'{card_val}{suit}')
                if card in top:
                    break
            else:
                top.append(Card.new(f'{card_val}{suits[len(top) - 1]}'))
            card_val += 1
        return top
    
    @classmethod
    def _get_row_rank(self, row):
        rank = evaluator.evaluate(row[:3], row[3:])
        rank_class = evaluator.class_to_string(evaluator.get_rank_class(rank))
        return rank, rank_class

    @classmethod
    def _hand_comp(self, hand1, hand2):
        rank1, _ = self._get_row_rank(hand1)
        rank2, _ = self._get_row_rank(hand2)
        return rank2 - rank1
    
    @classmethod
    def _top_middle_valid(self, hand):
        top, middle = hand[2].copy(), hand[1]
        top = self._extend_top(top)
        top_rank, top_class = self._get_row_rank(top)
        middle_rank, _ = self._get_row_rank(middle)
        return (top_rank >= middle_rank or top_class == "Straight")

    @classmethod
    def _middle_bottom_valid(self, hand):
        middle, bottom = hand[1], hand[0]
        return (self._hand_comp(middle, bottom) <= 0)

    @classmethod
    def _is_foul(self, hand):
        return not (self._top_middle_valid(hand) and self._middle_bottom_valid(hand))
    
    '''
    Returns the royalty for a given row using the provided royalty map.
    For the top row (with less than 5 cards), it checks various combos.
    '''
    @classmethod
    def _row_royalty(self, royalty_map, row):
        if len(row) < 5:
            row_str = "".join([Card.int_to_str(card)[0] for card in row])
            combos = [row_str, row_str[:2], row_str[1:]]
            for combo in combos:
                if combo in royalty_map:
                    return royalty_map[combo]
            return 0
        else:
            _, rank_class = self._get_row_rank(row)
            if rank_class in royalty_map:
                return royalty_map[rank_class]
            return 0
    
    '''
    Computes and returns the total royalty for a hand (agent or environment)
    by summing the royalties of the bottom, middle, and top rows.
    '''
    @classmethod
    def _total_royalty(self, hand):
        bottom, middle, top = hand
        return (self._row_royalty(bottom_royalty_map, bottom) +
                self._row_royalty(middle_royalty_map, middle) +
                self._row_royalty(top_royalty_map, top))

    '''
    Returns whether the player qualifies for fantasyland in the next hand,
    based on their current finished hand and previous fantasy status.
    '''
    @classmethod
    def _is_fantasyland(self, hand, is_fantasy):
        if self._is_foul(hand):
            return False
        bottom, middle, top = hand[0], hand[1], self._extend_top(deepcopy(hand[2]))
        if not is_fantasy:
            worst = [Card.new('Qh'), Card.new('Qd'), Card.new('2s'), Card.new('3c'), Card.new('4h')]
            _, rank_str = self._get_row_rank(top)
            return (self._hand_comp(top, worst) >= 0 and rank_str != "Straight")
        else:
            bottom_worst = [Card.new('2h'), Card.new('2d'), Card.new('2s'), Card.new('2c'), Card.new('3h')]
            middle_worst = [Card.new('2h'), Card.new('2d'), Card.new('2s'), Card.new('3c'), Card.new('3h')]
            top_worst = [Card.new('2h'), Card.new('2d'), Card.new('2s'), Card.new('3c'), Card.new('4h')]
            return (
                self._hand_comp(top, top_worst) >= 0
                or self._hand_comp(middle, middle_worst) >= 0
                or self._hand_comp(bottom, bottom_worst) >= 0
            )
  
    '''
    Computes the reward both agents get at the end of the hand using the simple
    total reward formula from Tan & Xiao (2018).
    '''
    @classmethod
    def _get_reward(self, hand1, hand2):
        # If both foul immediately return 0
        if self._is_foul(hand1) and self._is_foul(hand2):
            return 0, 0
            
        # Otherwise at most 1 fouled
        hand1[2] = self._extend_top(hand1[2])
        hand2[2] = self._extend_top(hand2[2])
        reward1, reward2 = 0, 0
        for row1, row2 in zip(hand1, hand2):
            if self._is_foul(hand1):
                reward1 -= 1
                reward2 += 1  
            elif self._is_foul(hand2):
                reward1 += 1
                reward2 -= 1
            # Otherwise compare rows, accounting for possibility of equality
            else:
                add1 = (
                    1 if self._hand_comp(row1, row2) > 0 
                    else (-1 if self._hand_comp(row1, row2) < 0 else 0)
                )
                add2 = -add1
                reward1 += add1
                reward2 += add2
        # Scoop!
        if reward1 == 3:
            reward1 += 3
            reward2 -= 3
        elif reward2 == 3:
            reward1 -= 3
            reward2 += 3

        # Royalties!
        royalty1 = 0 if self._is_foul(hand1) else self._total_royalty(hand1)
        royalty2 = 0 if self._is_foul(hand2) else self._total_royalty(hand2)

        # Return plugged-in formula
        return reward1 + (royalty1 - royalty2), reward2 + (royalty2 - royalty1)