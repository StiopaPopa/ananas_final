
import numpy as np
import json
from pineapple_utils import PineappleUtils as P
from treys import Deck, Card

SAVE_FILE = "expert_first5_data.json"

def generate_random_hand():
    """Generates a random 5-card hand (0-51 int representations)."""
    deck = Deck()
    return [int(P._treys_to_int(deck.draw(1)[0])) for _ in range(5)]

def show_hand(hand):
    """Displays a 5-card hand in a readable format."""
    print("\nYour hand: ", [Card.int_to_pretty_str(P._int_to_treys(card)) for card in hand])

def collect_expert_label():
    """Generate a hand, ask user to label the best placement, and save."""
    hand = generate_random_hand()
    show_hand(hand)

    while True:
        try:
            placement = input("\nEnter placement as 5 numbers (0=Bottom, 1=Middle, 2=Top), e.g., '0 1 2 0 1': ").strip()
            placement = [int(x) for x in placement.split()]
            if len(placement) == 5 and all(x in [0, 1, 2] for x in placement):
                break
            print("Invalid input. Enter exactly 5 numbers (0,1,2).")
        except ValueError:
            print("Invalid format. Enter numbers separated by spaces.")

    # Save data
    data = {"hand": hand, "placement": placement}
    
    try:
        with open(SAVE_FILE, "r") as f:
            all_data = json.load(f)
    except FileNotFoundError:
        all_data = []

    all_data.append(data)
    print(f"# expert placements collected: {len(all_data)}")

    with open(SAVE_FILE, "w") as f:
        json.dump(all_data, f, indent=4)

    print("Placement saved!")

if __name__ == "__main__":
    while True:
        collect_expert_label()
        cont = input("Collect another? (y/n): ").strip().lower()
        if cont != "y":
            break
