
import gym
from gym import Env
from gridworld import *
from core import Agent
from agents import ApproxQAgent

def main():

    env = gym.make("PuckWorld-v0")
    directory = "/home/qiang/workspace/reinforce/reinforce/monitor"
    
    env = gym.wrappers.Monitor(env, directory, force=True)
    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Box):
        input_dim = env.observation_space.shape[0]
    else:
        input_dim = 1

    agent = ApproxQAgent(env = env,
                         input_dim = input_dim,
                         output_dim = env.action_space.n,
                         hidden_dim = 100)
    print("Learning...")  
    agent.learning(gamma=0.9, 
                   alpha=0.1, 
                   max_episodes = 1000)


if __name__ == "__main__":
    main()