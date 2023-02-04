import random

from model_free.model_free import ModelFree

VALUES = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
SUITS = ['♠','♥','♦','♣']

CARDS = [(value,suit) for value in VALUES for suit in SUITS]

states = [(i, False, dealer_showing) 
    for i in range(4,22) 
    for dealer_showing in VALUES]

states += [(i, True, dealer_showing) 
    for i in range(12,22)
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
    if action == 'hit':
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
        if player_sum > 21:
            return (state, -1), True
        else:
            new_state = (player_sum, usable_ace, dealer_showing)
            return (new_state, 0), False
    
    dealer_cards = [dealer_showing]
    dealer_sum = count(dealer_cards)
    if action == 'stand':
        dealer_plays = True
        i=0
        while dealer_plays:
            dealer_sum = count(dealer_cards)
            if dealer_sum < 17:
                dealer_cards.append(random.choice(VALUES))
                continue
            elif dealer_sum > 21:
                return (state, 1), True
            elif 17 <= dealer_sum < 22:
                dealer_plays = False
    
    if dealer_sum > player_sum:
        return (state, -1), True
    elif dealer_sum < player_sum:
        return (state, 1), True
    elif dealer_sum == player_sum:
        return (state, 0), True


black_jack = ModelFree(
    states=states,
    actions=actions,
    gamma=0.9,
    transition=black_jack_transition    
)

state_0 = random.choice(states)
action_0 = random.choice(actions)


black_jack.generate_episode(
    state_0 = state_0,
    action_0 = action_0  
)

black_jack.vq_pi()