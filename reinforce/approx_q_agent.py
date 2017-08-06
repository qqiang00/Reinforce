from random import random, choice
from gym import Env
import gym
from gridworld import *
from core import Transition, Experience, Agent
from approximator import Approximator
from agents import ApproxQAgent
import torch


def testApproxQAgent():
    env = gym.make("MountainCar-v0")
    #env = SimpleGridWorld()
    directory = "/home/qiang/workspace/reinforce/monitor"
    
    env = gym.wrappers.Monitor(env, directory, force=True)
    agent = ApproxQAgent(env,
                         trans_capacity = 10000,    # 记忆容量（按状态转换数计）
                         hidden_dim = 16)           # 隐藏神经元数量
    env.reset()
    print("Learning...")  
    agent.learning(gamma=0.99,          # 衰减引子
                   learning_rate = 1e-3,# 学习率
                   batch_size = 64,     # 集中学习的规模
                   max_episodes=2000,   # 最大训练Episode数量
                   min_epsilon = 0.01,   # 最小Epsilon
                   epsilon_factor = 0.3,# 开始使用最小Epsilon时Episode的序号占最大
                                        # Episodes序号之比，该比值越小，表示使用
                                        # min_epsilon的episode越多
                   epochs = 2           # 每个batch_size训练的次数
                   )


if __name__ == "__main__":
    testApproxQAgent()