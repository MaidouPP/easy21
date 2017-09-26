import random
import numpy as np
import copy
from Utility.utilities import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

DEALER_VALUES = 10
PLAYER_VALUES = 21
ACT_VALUES = 2  # stick or hit

class Agent:
    def __init__(self):
        self.value = np.zeros((DEALER_VALUES, PLAYER_VALUES))
        self.iter = 0
        self.method = ""
        self.n0 = 0  # to determine

        # q-value table
        self.Q = np.zeros((DEALER_VALUES, PLAYER_VALUES, ACT_VALUES))
        # eligibility trace value
        self.E = np.zeros((DEALER_VALUES, PLAYER_VALUES, ACT_VALUES))
        # visit times
        self.N = np.zeros((DEALER_VALUES, PLAYER_VALUES, ACT_VALUES))
        # value function
        self.V = np.zeros((DEALER_VALUES, PLAYER_VALUES))

        self.wins = 0

    def eps_greedy_action(self, state):
        visit = sum(self.N[state.dealer-1, state.playerSum-1, :])
        eps = float(self.n0) / (self.n0 + visit)
        r = random.random()
        if r < eps:
            return Action(0) if random.random() < 0.5 else Action(1)
        else:
            return Action(np.argmax(self.Q[state.dealer-1, state.playerSum-1, :]))  # pick the optimal policy

    def MC_control(self, iterations, n0):
        self.iter = iterations
        self.method = "MC_control"
        self.n0 = n0
        history = []  # for mote-carlo update

        for ep in xrange(self.iter):
            history = []
            dealerCard = random.randint(1, 10)
            playerCard = random.randint(1, 10)
            state = State(dealer=dealerCard, playerSum=playerCard)

            # monte-carlo can only update action-value after each episode ends!
            while state.isTerminal == False:
                act = self.eps_greedy_action(state)
                history.append((state, act))
                self.N[state.dealer-1, state.playerSum-1, act.value] += 1
                state = step(state, act)

            for st, ac in history:
                tmp = float(1) / self.N[st.dealer-1, st.playerSum-1, ac.value]
                err = calculate_reward(state) - self.Q[st.dealer-1, st.playerSum-1, ac.value]
                # print calculate_reward(st), " ", self.Q[st.dealer-1, st.playerSum-1, ac.value]
                # print calculate_reward(state), " ", self.Q[st.dealer-1, st.playerSum-1, ac.value]
                self.Q[st.dealer-1, st.playerSum-1, ac.value] += float(tmp) * err

            if calculate_reward(st) == 1 :
                self.wins += 1

            if ep%200 == 0:
                print "episode... %s" % ep

        for d in xrange(DEALER_VALUES):
            for p in xrange(PLAYER_VALUES):
                self.V[d, p] = float(max(self.Q[d, p, :]))
        print "win rate: %s" % (float(self.wins) / self.iter)

    # Sarsa(lambda) method
    def TD_control(self, iterations, l):
        self.iter = iterations
        self.method = "Sarsa"

        for ep in xrange(self.iter):
            self.E = np.zeros((DEALER_VALUES, PLAYER_VALUES, ACT_VALUES))

            dealerCard = random.randint(1, 10)
            playerCard = random.randint(1, 10)
            # initialize s
            state = State(dealer=dealerCard, playerSum=playerCard)
            # initialize a using e-greedy
            act = self.eps_greedy_action(state)

            while state.isTerminal == False:
                self.N[state.dealerSum-1, state.playerSum-1, act.value] += 1
                newState = step(state, act)
                reward = calculate_reward(newState)
                if newState.isTerminal == False:
                    newAct = self.eps_greedy_action(newState)
                else:
                    # it doesn't matter what action to take
                    newAct = Action(0)

                try:
                    error = reward + self.Q[newState.dealerSum-1, newState.playerSum-1, newAct.value] - self.Q[state.dealerSum-1, state.playerSum-1, act.value]
                except:
                    error = reward - self.Q[state.dealerSum-1, state.playerSum-1, act.value]
                self.E[state.dealerSum-1, state.playerSum-1, act.value] += 1
                # update step
                alpha = float(1) / self.N[state.dealerSum-1, state.playerSum-1, act.value]

                for dl in xrange(DEALER_VALUES):
                    for pl in xrange(PLAYER_VALUES):
                        for ac in xrange(ACT_VALUES):
                            self.Q[dl, pl, ac] += alpha * error * self.E[dl, pl, ac]
                            self.E[dl, pl, ac] = self.E[dl, pl, ac] * l

                state = newState
                act = newAct

            if calculate_reward(state)==1:
                self.wins += 1

            if ep%200 == 0:
                print "episode... %s" % ep

            for d in xrange(DEALER_VALUES):
                for p in xrange(PLAYER_VALUES):
                    self.V[d, p] = float(max(self.Q[d, p, :]))

        print "win rate: %s" % (float(self.wins) / self.iter)


    def plot_state(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        X = np.arange(0, DEALER_VALUES, 1, int)
        Y = np.arange(0, PLAYER_VALUES, 1, int)
        X, Y = np.meshgrid(X, Y)
        Z = self.V[X, Y]
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
        plt.show()