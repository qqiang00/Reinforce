
import gym
from gym import Env
from .gridworld import *
from .core import Agent

def main():

    env = gym.make("WindyGridWorld-v0")
    directory = "/home/qiang/workspace/reinforce/python/monitor"
    
    env = gym.wrappers.Monitor(env, directory, force=True)
    agent = ApproxQAgent(env)
    env.reset()
    print("Learning...")  
    agent.learning(gamma=0.9, 
                   alpha=0.1, 
                   max_episode_num=500)


if __name__ == "__main__":
    main()