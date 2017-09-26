import numpy as np
import copy

class Card:
    def __init__(self):
        self.value = np.random.randint(1, 10)
        self.isBlack = np.random.choice([True, False], size=1, p=[0.667, 0.333])

    def __eq__(self, other):
        return (self.value, self.isBlack) == (other.value, other.isBlack)

class State:
    def __init__(self, dealer, playerSum, isTerminal=False):
        self.isTerminal = isTerminal
        self.dealer = dealer  # the first card of dealer
        self.playerSum = playerSum
        self.dealerSum = dealer  # the final point of dealer
        self.burst = False
        self.win = 0  # 0: dealer, 1: player

    def copy(self):
        return State(self.dealer, self.playerSum, self.isTerminal)

class Action:
    # action is hit with value zero
    # action is stick (stop get cards) with value one
    def __init__(self, action):
        self.value = action

def dealer_score(state):
    sum = state.dealer
    while sum < 17:
        card = Card()
        sum += card.value if card.isBlack else -card.value
    # print sum
    state.dealerSum = sum

def step(state, action):
    if state.isTerminal:
        Exception('step on terminated state')
    newState = copy.deepcopy(state)
    if action.value == 0:  # hit
        card = Card();
        newState.playerSum += card.value if card.isBlack else -card.value
        if newState.playerSum > 21 or newState.playerSum < 0:
            newState.isTerminal = True
            newState.burst = True
        elif newState.playerSum == 21:
            newState.isTerminal = True
    else:
        newState.isTerminal = True
        dealer_score(newState)
    return newState

def calculate_reward(state):
    if state.isTerminal == False:
        return 0
    if state.burst:
        return -1
    else:
        if 0 < state.dealerSum < 22:
            if state.dealerSum > state.playerSum:
                return -1
            elif state.dealerSum == state.playerSum:
                return 0
            else:
                state.win = 1
                return 1
        else:
            state.win = 1
            return 1