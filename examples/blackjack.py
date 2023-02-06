import random

from rl.solvers import (
    alpha_mc,
    off_policy_mc,
    tdn
)

VALUES = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
SUITS = ['♠','♥','♦','♣']
CARDS = [(value,suit) for value in VALUES for suit in SUITS]

states = [(i, False, dealer_showing) 
    for i in range(4,21) 
    for dealer_showing in VALUES]

states += [(i, True, dealer_showing) 
    for i in range(12,21)
    for dealer_showing in VALUES]

actions = ['hit', 'stand']

def count(cards):
    counts = [0]
    for value in cards:
        if value in ['J','Q','K']:
            counts = [c+10 for c in counts]
        elif value == 'A':
            counts = [c+1 for c in counts] + [c+11 for c in counts]
        else:
            counts = [c+int(value) for c in counts]

    valid_counts = [c for c in counts if c <= 21]
    if len(valid_counts) == 0:
        return min(counts)
    return max(valid_counts)


def black_jack_transition(state, action):
    player_sum, usable_ace, dealer_showing = state
    
    if action == 'hit' and player_sum < 21:
        new_card = random.choice(VALUES)
        if new_card == 'A':
            if player_sum + 11 > 21:
                card_value = 1 
            else:
                card_value = 11
                usable_ace = True
        elif new_card in ['J','Q','K']:
            card_value = 10
        else:
            card_value = int(new_card)

        player_sum += card_value
        if usable_ace and player_sum > 21:
            player_sum -= 10
            usable_ace = False

        if player_sum > 21:
            return (state, -1.), True
        elif player_sum == 21:
            pass
        else:
            new_state = (player_sum, usable_ace, dealer_showing)
            return (new_state, 0.), False
    
    dealer_cards = [dealer_showing]
    dealer_sum = count(dealer_cards)
    if action == 'stand':
        dealer_plays = True
        while dealer_plays:
            dealer_sum = count(dealer_cards)
            if dealer_sum < 17:
                dealer_cards.append(random.choice(VALUES))
                continue
            elif dealer_sum > 21:
                return (state, 1.), True
            elif 17 <= dealer_sum < 22:
                dealer_plays = False
    
    if dealer_sum > player_sum:
        return (state, -1.), True
    elif dealer_sum < player_sum:
        return (state, 1.), True
    elif dealer_sum == player_sum:
        return (state, 0.), True


vqpi, samples = alpha_mc(states, actions, black_jack_transition, gamma=0.9,
    use_N=True, n_episodes=1E4, first_visit=False)

 