from Agent.agent import Agent
import matplotlib.pyplot as plt

def test_mc_control(iter=50000, n0=100):
    agent = Agent()
    agent.MC_control(iter, n0)
    agent.plot_state()

def test_td_control(iter=50000):
    lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for l in lambdas:
        agent = Agent()
        agent.TD_control(iter, l)
        agent.plot_state()

def test_td_control_linear_app(iter=50000):
    lambdas = [0.0, 1.0]
    for l in lambdas:
        agent = Agent()
        agent.TD_control_linear_app(iter, 0.9)
        agent.plot_state()

if __name__=='__main__':
    # test_mc_control()
    # test_td_control()
    test_td_control_linear_app()