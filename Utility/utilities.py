import numpy as np

class Card:
    def __init__(self):
        self.value = np.random.randint(low=1, high=10, size=1)
        self.isBlack = np.random.randint([True, False], size=1, p=[0.667, 0.333])

    def __eq__(self, other):
        return (self.value, self.isBlack) == (other.value, other.isBlack)

class State:
    def __init__(self, dealerSum, playerSum, isTerminal=False):
        self.isTerminal = isTerminal
        self.dealerSum = dealerSum
        self.playerSum = playerSum
        self.burst = False

    def copy(self):
        return State(self.dealerSum, playerSum, isTerminal)

class Action:
    # action is hit with value zero
    # action is stick (stop get cards) with value one
    def __init__(self, action):
        self.value = action

def dealerPlay(state):
    sum = state.dealerSum
    while 0 < sum < 17:
        card = Card()
        sum += card.value if card.isBlack else -card.value
    state.dealerSum = sum

def step(state, action):
    if state.isTerminal:
        Exception('step on terminated state')
    if action.value == 0:  # hit
        card = Card();
        state.playerSum += card.value if card.isBlack else -card.value
        if state.playerSum > 21 or state.playerSum < 0:
            state.isTerminal = True
            state.burst = True
    else:
        state.isTerminal = True
        dealerPlay(state)
    return state

def caculateReward(state):
    if state.isTerminal==False:  # if the player hasn't stopped
        return 0
    else:
        if state.burst:
            return -1
        else:
            if 0 < state.dealerSum < 22:
                if state.dealerSum > state.playerSum:
                    return -1
                elif state.dealerSum == state.playerSum:
                    return 0
                else:
                    return 1
            else:
                return 1