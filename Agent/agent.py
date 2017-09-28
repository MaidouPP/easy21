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
        # td-contorl linear approximator theta
        self.theta = np.zeros(36)
        # td-control linear approximation evaluation trace
        self.E_app = np.zeros(36)

        self.wins = 0

    def eps_greedy_action(self, state):
        visit = sum(self.N[state.dealer-1, state.playerSum-1, :])
        eps = float(self.n0) / (self.n0 + visit)
        r = random.random()
        if r < eps:
            return Action(0) if random.random() < eps/2 else Action(1)
        else:
            return Action(np.argmax(self.Q[state.dealer-1, state.playerSum-1, :]))  # pick the optimal policy

    def feature_vec(self, state, action, d, p):
        feature = []
        for i in xrange(len(d)):
            for j in xrange(len(p)):
                if p[j][0] <= state.playerSum <= p[j][1] and d[i][0] <= state.dealerSum <= d[i][1]:
                    feature.append(1)
                else:
                    feature.append(0)
        if action.value==0:
            return np.concatenate([feature, np.zeros(18)])
        else:
            return np.concatenate([np.zeros(18), feature])

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

    def TD_control_linear_app(self, iterations, lamb, eps = 0.05, alpha = 0.01):
        self.iter = iterations
        self.method = "td_control_linear_approximation"
        self.theta = np.random.random(36)*0.5
        dSeg = [[1,4],[4,7],[7,10]]
        pSeg = [[1,6],[4,9],[7,12],[10,15],[13,18],[16,21]]
        self.wins = 0

        for ep in xrange(self.iter):
            self.E_app = np.zeros(36)

            dealerCard = random.randint(1, 10)
            playerCard = random.randint(1, 10)
            # initialize s
            state = State(dealer=dealerCard, playerSum=playerCard)
            act = None
            feature = None

            r = np.random.random()
            if r < eps:
                act = Action(0) if random.random() < 0.5 else Action(1)
                feature = self.feature_vec(state, act, dSeg, pSeg)
                qValueApp = sum(self.theta * feature)
            else:
                qValueApp = -10000
                act = Action(0)
                feature = []
                for a in xrange(ACT_VALUES):
                    f = self.feature_vec(state, Action(a), dSeg, pSeg)
                    tmp = sum(self.theta * f)
                    if tmp>qValueApp:
                        qValueApp = tmp
                        act = Action(a)
                        feature = f

            while state.isTerminal == False:
                # pay attention to how E is updated!!
                self.E_app += feature
                newState = step(state, act)
                reward = calculate_reward(newState)
                currF = sum(self.theta * feature)
                currX = feature

                # update parameters
                r = np.random.random()
                if r < eps:
                    act = Action(0) if random.random() < 0.5 else Action(1)
                    feature = self.feature_vec(state, act, dSeg, pSeg)
                    qValueApp = sum(self.theta * feature)
                else:
                    qValueApp = -10000
                    act = Action(0)
                    feature = []
                    for a in xrange(ACT_VALUES):
                        f = self.feature_vec(state, Action(a), dSeg, pSeg)
                        tmp = sum(self.theta * f)
                        if tmp > qValueApp:
                            qValueApp = tmp
                            act = Action(a)
                            feature = f

                # update parameters
                error = reward + sum(self.theta * feature) - currF
                self.theta += alpha * error * self.E_app
                self.E_app = lamb * self.E_app

                state = newState

            if calculate_reward(state)==1:
                self.wins += 1

            if ep%200 == 0:
                print "episode... %s wining rate... %s" % (ep, float(self.wins)/(ep+1))

        for d in xrange(DEALER_VALUES):
            for p in xrange(PLAYER_VALUES):
                for a in xrange(ACT_VALUES):
                    self.Q[d, p, a] = sum(self.theta * self.feature_vec(State(d, p), Action(a), dSeg, pSeg))
                self.V[d, p] = max(self.Q[d, p, :])

    def plot_state(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        X = np.arange(0, DEALER_VALUES, 1, int)
        Y = np.arange(0, PLAYER_VALUES, 1, int)
        X, Y = np.meshgrid(X, Y)
        Z = self.V[X, Y]
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
        plt.show()