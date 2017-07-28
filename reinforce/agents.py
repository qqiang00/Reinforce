#!/home/qiang/PythonEnv/venv/bin/python3.5
# -*- coding: utf-8 -*-
# agents of reinforcment learning

# Author: Qiang Ye
# Date: July 27, 2017

from random import random, choice
from gym import Env
import gym
from gridworld import *
from core import Transition, Experience, Agent
from func_approximator import NeuralNetwork


class SarsaAgent(Agent):
    def __init__(self, env:Env, capacity:int = 20000):
        super(SarsaAgent, self).__init__(env, capacity)
        #Agent.__init__(self, env, cap)
        # 保存一些Agent可以观测到的环境信息以及已经学到的经验
        self.Q = {}  # {s0:[,,,,,,],s1:[,,,,,]} 数组内元素个数为行为空间大小
        self.resetAgent()
        return

    def resetAgent(self):
        self.state = self.env.reset()
        s_name = self._get_state_name(self.state)
        self._assert_state_in_Q(s_name, randomized = False)
    
    # using simple decaying epsilon greedy exploration
    def _curPolicy(self, s, episode_num, use_epsilon):
        epsilon = 1.00 / episode_num
        Q_s = self.Q[s]
        str_act = "unknown"
        rand_value = random()
        action = None
        if use_epsilon and rand_value < epsilon:  
            action = self.env.action_space.sample()
        else:
            str_act = max(Q_s, key=Q_s.get)
            action = int(str_act)
        return action

    # Agent依据当前策略和状态决定下一步的动作
    def performPolicy(self, s, episode_num, use_epsilon=True):
        return self._curPolicy(s, episode_num, use_epsilon)

    # sarsa learning
    def sarsaLearning(self, gamma, alpha, max_episode_num):
        # self.Position_t_name, self.reward_t1 = self.observe(env)
        total_time = 0
        time_in_episode = 0
        num_episode = 1
        while num_episode <= max_episode_num:
            self.state = str(self.env.reset())
            s0 = self.state
            self.env.render()
            a0 = self.performPolicy(s0, num_episode)
            # print(self.action_t.name)
            time_in_episode = 0
            is_done = False
            while not is_done:
                # add code here
                s1, r1, is_done, info, total_reward = self.act(a0)
                self.env.render()
                s1 = str(s1)
                self._assert_state_in_Q(s1, randomized = True)
                # 在下行代码中添加参数use_epsilon = False即变城Q学习算法
                a1 = self.performPolicy(s1, num_episode, use_epsilon=True)
                old_q = self._get_Q(s0, a0)
                q_prime = self._get_Q(s1, a1)
                td_target = r1 + gamma * q_prime
                #alpha = alpha / num_episode
                new_q = old_q + alpha * (td_target - old_q)
                self._set_Q(s0, a0, new_q)
                s0, a0 = s1, a1
                time_in_episode += 1

            print(self.experience.last)
            
            total_time += time_in_episode
            num_episode += 1
        self.experience.last.print_detail()
        return


    def _is_state_in_Q(self, s):
        return self.Q.get(s) is not None

    def _init_state_value(self, s_name, randomized = True):
        if not self._is_state_in_Q(s_name):
            self.Q[s_name] = {}
            for action in range(self.action_space.n):
                default_v = random() / 10 if randomized is True else 0.0
                self.Q[s_name][action] = default_v

    def _assert_state_in_Q(self, s, randomized=True):
        # 　cann't find the state
        if not self._is_state_in_Q(s):
            self._init_state_value(s, randomized)
    
    def _get_state_name(self, state):   # 得到状态对应的字符串作为以字典存储的价值函数
        return str(state)               # 的键值，应针对不同的状态值单独设计，避免重复
                                        # 这里仅针对格子世界
        

    def _get_Q(self, s, a):
        self._assert_state_in_Q(s, randomized=True)
        return self.Q[s][a]

    def _set_Q(self, s, a, value):
        self._assert_state_in_Q(s, randomized=True)
        self.Q[s][a] = value



class SarsaLambdaAgent(Agent):
    def __init__(self, env:Env, cap: 0):
        super(SarsaLambdaAgent, self).__init__(env, cap)
        #Agent.__init__(self, env, cap)
        # 保存一些Agent可以观测到的环境信息以及已经学到的经验
        self.Q = {}  # {s0:[,,,,,,],s1:[]} 数组内元素个数为行为空间大小
        self.E = {}  # Elegibility Trace
        self.resetAgent()
        return

    def resetAgent(self):
        self.state = self.env.reset()
        s_name = self._get_state_name(self.state)
        self._assert_state_in_QE(s_name, randomized = False)
    
    # using simple decaying epsilon greedy exploration
    def _curPolicy(self, s, episode_num, use_epsilon):
        epsilon = 1.00 / (10*episode_num)
        Q_s = self.Q[s]
        str_act = "unknown"
        rand_value = random()
        action = None
        if use_epsilon and rand_value < epsilon:  
            action = self.env.action_space.sample()
        else:
            str_act = max(Q_s, key=Q_s.get)
            action = int(str_act)
        return action

    # Agent依据当前策略和状态决定下一步的动作
    def performPolicy(self, s, episode_num, use_epsilon=True):
        return self._curPolicy(s, episode_num, use_epsilon)

    def sarsaLambdaLearning(self, lambda_, gamma, alpha, max_episode_num):
        total_time = 0
        time_in_episode = 0
        num_episode = 1
        while num_episode <= max_episode_num:
            self._resetEValue()
            s0 = str(self.env.reset())
            self.env.render()
            a0 = self.performPolicy(s0, num_episode)
            # print(self.action_t.name)
            time_in_episode = 0
            is_done = False
            while not is_done:
                # add code here
                s1, r1, is_done, info = self.act(a0)
                self.env.render()
                s1 = str(s1)
                self._assert_state_in_QE(s1, randomized = True)
                # 在下行代码中添加参数use_epsilon = False即变成Q学习算法
                a1= self.performPolicy(s1, num_episode)

                q = self._get_(self.Q, s0, a0)
                q_prime = self._get_(self.Q, s1, a1)
                delta = r1 + gamma * q_prime - q

                e = self._get_(self.E, s0,a0)
                e = e + 1
                self._set_(self.E, s0, a0, e)

                state_action_list = list(zip(self.E.keys(),self.E.values()))
                for s, a_es in state_action_list:
                    for a in range(self.action_space.n):
                        e_value = a_es[a]
                        old_q = self._get_(self.Q, s, a)
                        new_q = old_q + alpha * delta * e_value
                        new_e = gamma * lambda_ * e_value
                        self._set_(self.Q, s, a, new_q)
                        self._set_(self.E, s, a, new_e)

                if num_episode == max_episode_num:
                    # print current action series
                    print("t:{0:>2}: s:{1}, a:{2:10}, s1:{3}".
                          format(time_in_episode, s0, a0, s1))
                
                s0, a0 = s1, a1
                time_in_episode += 1

            print("Episode {0} takes {1} steps.".format(
                num_episode, time_in_episode))
            total_time += time_in_episode
            num_episode += 1
        return


    def _is_state_in_Q(self, s):
        return self.Q.get(s) is not None

    def _init_state_value(self, s_name, randomized = True):
        if not self._is_state_in_Q(s_name):
            self.Q[s_name], self.E[s_name] = {},{}
            for action in range(self.action_space.n):
                default_v = random() / 10 if randomized is True else 0.0
                self.Q[s_name][action] = default_v
                self.E[s_name][action] = 0.0

    def _assert_state_in_QE(self, s, randomized=True):
        if not self._is_state_in_Q(s):
            self._init_state_value(s, randomized)
    
    def _get_state_name(self, state):   # 得到状态对应的字符串作为以字典存储的价值函数
        return str(state)               # 的键值，应针对不同的状态值单独设计，
                                        # 这里仅针对格子世界
        
    def _get_(self, QorE, s, a):
        self._assert_state_in_QE(s, randomized=True)
        return QorE[s][a]

    def _set_(self, QorE, s, a, value):
        self._assert_state_in_QE(s, randomized=True)
        QorE[s][a] = value

    def _resetEValue(self):
        for value_dic in self.E.values():
            for action in range(self.action_space.n):
                value_dic[action] = 0.00


class ApproxQAgent(Agent):
    def __init__(self, env:Env = None,
                       trans_capacity = 20000,
                       input_dim:int,
                       output_dim:int):
        super(DQN, self).__init__(env, trans_capacity)
        self.input_dim = input_dim
        self.input_dim = output_dim
        self.Q = NeuralNetwork(input_dim = input_size,
                                       output_size,
                                       hidden_dim = 10)
        self.policy_value_fn = copy(self.value_fn1) # 更新参数

    def _curPolicy(self, s, episode_num, use_epsilon):
        '''依据更新策略的价值函数(网络)产生一个行为
        '''
        epsilon = 1.00 / (episode_num+1)
        Q_s = self.policy_value_fn(s)
        str_act = "unknown"
        rand_value = random()
        action = None
        if use_epsilon and rand_value < epsilon:  
            action = self.env.action_space.sample()
        else:
            str_act = max(Q_s, key=Q_s.get)
            action = int(str_act)
        return action
        
    def performPolicy(self, s, episode_num, use_epsilon=True):
        return sefl.curPolicy(s, episode_num, use_episode)

    def _get_Q(self, s, a):
        '''根据目标价值网络得到一个状态行为价值
        '''
        return self.target_value_fn(s)[a]

    def _update_Q_net(self):
        self.Q = copy(self.policy_value_fn)

    def learning(self):
        # self.Position_t_name, self.reward_t1 = self.observe(env)
        total_time, time_in_episode, num_episode = 0, 0, 0

        while num_episode < max_episode_num:
            s0 = self.env.reset()
            self.env.render()
            a0 = self.performPolicy(s0, num_episode, use_epsilon = True)
            
            time_in_episode = 0
            is_done = False
            while not is_done:
                # a0 = self.performPolicy(s0, num_episode)
                s1, r1, is_done, info = self.act(a0)
                self.env.render()
                # 在下行代码中添加参数use_epsilon = False即变成Q学习算法
                a1 = self.performPolicy(s1, num_episode, use_epsilon=True)

                old_q = self._get_Q(s0, a0)
                q_prime = self._get_Q(s1, a1)
                td_target = r1 + gamma * q_prime
                #alpha = alpha / num_episode
                new_q = old_q + alpha * (td_target - old_q)
                self.Q.fit(old_q, new_q)
                
                if num_episode == max_episode_num:
                    print("t:{0:>2}: s:{1}, a:{2:2}, s1:{3}".\
                        format(time_in_episode, s0, a0, s1))

                s0, a0 = s1, a1
                time_in_episode += 1
                if num_episode %10 == 0:
                    self._update_Q_net()

            print("Episode {0} takes {1} steps.".format(
                num_episode, time_in_episode))
            total_time += time_in_episode
            num_episode += 1
        return   
