from Agent.agent import Agent
import matplotlib.pyplot as plt

def test_mc_control(iter=50000, n0=100):
    agent = Agent()
    agent.MC_control(iter, n0)
    agent.plot_state()

if __name__=='__main__':
    test_mc_control()
