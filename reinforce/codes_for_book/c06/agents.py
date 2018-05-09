#!/home/qiang/PythonEnv/venv/bin/python3.5
# -*- coding: utf-8 -*-
# agents of reinforcment learning

# Author: Qiang Ye
# Date: July 27, 2017

from random import random, choice
from gym import Env, spaces
import gym
#from gridworld import *
import numpy as np
from core import Transition, Experience, Agent
from utils import str_key, set_dict, get_dict
from utils import epsilon_greedy_pi, epsilon_greedy_policy
from utils import greedy_policy, learning_curve
from approximator import NetApproximator


class SarsaAgent(Agent):
    def __init__(self, env:Env, capacity:int = 20000):
        super(SarsaAgent, self).__init__(env, capacity)
        self.Q = {}

    def policy(self, A, s, Q, epsilon):
        return epsilon_greedy_policy(A, s, Q, epsilon)

    def learning_method(self, gamma = 0.9, alpha = 0.1, epsilon = 1e-5, display = False, lambda_ = None):
        self.state = self.env.reset()
        s0 = self.state
        if display:
            self.env.render()
        a0 = self.perform_policy(s0, epsilon)
        # print(self.action_t.name)
        time_in_episode, total_reward = 0, 0
        is_done = False
        while not is_done:
            # add code here
            s1, r1, is_done, info, total_reward = self.act(a0)
            if display:
                self.env.render()
            a1 = self.perform_policy(s1, epsilon)
            old_q = get_dict(self.Q, s0, a0)
            q_prime = get_dict(self.Q, s1, a1)
            td_target = r1 + gamma * q_prime
            #alpha = alpha / num_episode
            new_q = old_q + alpha * (td_target - old_q)
            set_dict(self.Q, new_q, s0, a0)
            s0, a0 = s1, a1
            time_in_episode += 1
        if display:
            print(self.experience.last_episode)
        return time_in_episode, total_reward
    

    
class SarsaLambdaAgent(Agent):
    def __init__(self, env:Env, capacity:int = 20000):
        super(SarsaLambdaAgent, self).__init__(env, capacity)
        self.Q = {}

    def policy(self, A, s, Q, epsilon):
        return epsilon_greedy_policy(A, s, Q, epsilon)

    def learning_method(self, lambda_ = 0.9, gamma = 0.9, alpha = 0.1, epsilon = 1e-5, display = False):
        self.state = self.env.reset()
        s0 = self.state
        if display:
            self.env.render()
        a0 = self.perform_policy(s0, epsilon)
        # print(self.action_t.name)
        time_in_episode, total_reward = 0, 0
        is_done = False
        E = {}
        while not is_done:
            # add code here
            s1, r1, is_done, info, total_reward = self.act(a0)
            if display:
                self.env.render()
            a1 = self.perform_policy(s1, epsilon)
            
            q = get_dict(self.Q, s0, a0)
            q_prime = get_dict(self.Q, s1, a1)
            delta = r1 + gamma * q_prime - q
            
            e = get_dict(E, s0, a0)
            e += 1
            set_dict(E, e, s0, a0)

            for s in self.S:
                for a in self.A:
                    e_value = get_dict(E, s, a)
                    old_q = get_dict(self.Q, s, a)
                    new_q = old_q + alpha * delta * e_value
                    new_e = gamma * lambda_ * e_value
                    set_dict(self.Q, new_q, s, a)
                    set_dict(E, new_e, s, a)
                    
            s0, a0 = s1, a1
            time_in_episode += 1
        if display:
            print(self.experience.last_episode)
        return time_in_episode, total_reward    
            
        
class QAgent(Agent):
    def __init__(self, env:Env, capacity:int = 20000):
        super(QAgent, self).__init__(env, capacity)
        self.Q = {}

    def policy(self, A, s, Q, epsilon):
        return epsilon_greedy_policy(A, s, Q, epsilon)
    
    def learning_method(self, gamma = 0.9, alpha = 0.1, epsilon = 1e-5, display = False, lambda_ = None):
        self.state = self.env.reset()
        s0 = self.state
        if display:
            self.env.render()
        # a0 = self.perform_policy(s0, epsilon)
        # print(self.action_t.name)
        time_in_episode, total_reward = 0, 0
        is_done = False
        while not is_done:
            # add code here
            a0 = self.perform_policy(s0, epsilon)
            s1, r1, is_done, info, total_reward = self.act(a0)
            if display:
                self.env.render()
            self.policy = greedy_policy
            a1 = greedy_policy(self.A, s1, self.Q)
            old_q = get_dict(self.Q, s0, a0)
            q_prime = get_dict(self.Q, s1, a1)
            td_target = r1 + gamma * q_prime
            #alpha = alpha / num_episode
            new_q = old_q + alpha * (td_target - old_q)
            set_dict(self.Q, new_q, s0, a0)
            # s0, a0 = s1, a1
            s0 = s1
            time_in_episode += 1
        if display:
            print(self.experience.last_episode)
        return time_in_episode, total_reward    
          
        
        
class DQNAgent(Agent):
    '''使用近似的价值函数实现的Q学习个体
    '''
    def __init__(self, env: Env = None,
                       capacity = 20000,
                       hidden_dim: int = 32,
                       batch_size = 128,
                       epochs = 2):
        if env is None:
            raise "agent should have an environment"
        super(DQNAgent, self).__init__(env, capacity)
        self.input_dim = env.observation_space.shape[0] # 状态连续
        self.output_dim = env.action_space.n # 行为离散
        # print("{},{}".format(self.input_dim, self.output_dim))
        self.hidden_dim = hidden_dim
        # 行为网络，该网络用来计算产生行为，以及对应的Q值，每次更新
        self.behavior_Q = NetApproximator(input_dim = self.input_dim,
                              output_dim = self.output_dim,
                              hidden_dim = self.hidden_dim)
        self.target_Q = self.behavior_Q.clone() # 计算价值目标的Q，不定期更新
        
        self.batch_size = batch_size  # 批学习一次状态转换数量
        self.epochs = epochs # 统一批状态转换学习的次数
        return
        
    def _update_target_Q(self):
        '''将更新策略的Q网络(连带其参数)复制给输出目标Q值的网络
        '''
        self.target_Q = self.behavior_Q.clone() # 更新计算价值目标的Q网络
        

    def policy(self, A, s, Q = None, epsilon = None):
        '''依据更新策略的价值函数(网络)产生一个行为
        '''
        Q_s = self.behavior_Q(s)
        rand_value = random()
        if epsilon is not None and rand_value < epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(Q_s))


    
    def _learn_from_memory(self, gamma, learning_rate):
        trans_pieces = self.sample(self.batch_size)  # 随机获取记忆里的Transmition
        states_0 = np.vstack([x.s0 for x in trans_pieces])
        actions_0 = np.array([x.a0 for x in trans_pieces])
        reward_1 = np.array([x.reward for x in trans_pieces])
        is_done = np.array([x.is_done for x in trans_pieces])
        states_1 = np.vstack([x.s1 for x in trans_pieces])

        X_batch = states_0
        y_batch = self.target_Q(states_0)  # 得到numpy格式的结果

        Q_target = reward_1 + gamma * np.max(self.target_Q(states_1), axis=1)*\
            (~ is_done) # is_done则Q_target==reward_1

        
        # switch this on will make DQN to DDQN    
        # 行为a'从行为价值网络中得到
        #a_prime = np.argmax(self.behavior_Q(states_1), axis=1).reshape(-1)
        # (s',a')的价值从目标价值网络中得到
        #Q_states_1 = self.target_Q(states_1)
        #temp_Q = Q_states_1[np.arange(len(Q_states_1)), a_prime]
        # (s,a)的目标价值根据贝尔曼方程得到
        #Q_target = reward_1 + gamma * temp_Q * (~ is_done) # is_done则Q_target==reward_1
        ## end of DDQN part
        
        y_batch[np.arange(len(X_batch)), actions_0] = Q_target
        
        # loss is a torch Variable with size of 1
        loss = self.behavior_Q.fit(x = X_batch, 
                           y = y_batch, 
                           learning_rate = learning_rate,
                           epochs = self.epochs)

        mean_loss = loss.sum().data[0] / self.batch_size
        self._update_target_Q()
        return mean_loss

    def learning_method(self, gamma = 0.9, alpha = 0.1, epsilon = 1e-5, 
                        display = False, lambda_ = None):
        self.state = self.env.reset()
        s0 = self.state
        if display:
            self.env.render()
        time_in_episode, total_reward = 0, 0
        is_done = False
        loss = 0
        while not is_done:
            # add code here
            s0 = self.state
            a0 = self.perform_policy(s0, epsilon)
            s1, r1, is_done, info, total_reward = self.act(a0)
            if display:
                self.env.render()
            
            if self.total_trans > self.batch_size:
                loss += self._learn_from_memory(gamma, alpha)
            # s0 = s1
            time_in_episode += 1
        loss /= time_in_episode
        if display:
            print("epsilon:{:3.2f},loss:{:3.2f},{}".format(epsilon,loss,self.experience.last_episode))
        return time_in_episode, total_reward  


