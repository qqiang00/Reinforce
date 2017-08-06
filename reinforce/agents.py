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
from approximator import Approximator
import torch


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
    def _curPolicy(self, s, num_episode, use_epsilon):
        epsilon = 1.00 / num_episode
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
    def performPolicy(self, s, num_episode, use_epsilon=True):
        return self._curPolicy(s, num_episode, use_epsilon)

    # sarsa learning
    def learning(self, gamma, alpha, max_num_episode):
        # self.Position_t_name, self.reward_t1 = self.observe(env)
        total_steps = 0
        step_in_episode = 0
        num_episode = 1
        while num_episode <= max_num_episode:
            self.state = str(self.env.reset())
            s0 = self.state
            self.env.render()
            a0 = self.performPolicy(s0, num_episode)
            # print(self.action_t.name)
            step_in_episode = 0
            is_done = False
            while not is_done:
                # add code here
                s1, r1, is_done, info, total_reward = self.act(a0)
                #self.env.render()
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
                step_in_episode += 1

            print(self.experience.last)
            
            total_steps += step_in_episode
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
        self._init_agent()
        return

    def _init_agent(self):
        self.state = self.env.reset()
        s_name = self._get_state_name(self.state)
        self._assert_state_in_QE(s_name, randomized = False)
    
    # using simple decaying epsilon greedy exploration
    def _curPolicy(self, s, num_episode, use_epsilon):
        epsilon = 1.00 / (10*num_episode)
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
    def performPolicy(self, s, num_episode, use_epsilon=True):
        return self._curPolicy(s, num_episode, use_epsilon)

    def learning(self, lambda_, gamma, alpha, max_num_episode):
        total_steps = 0
        step_in_episode = 0
        num_episode = 1
        while num_episode <= max_num_episode:
            self._resetEValue()
            s0 = str(self.env.reset())
            self.env.render()
            a0 = self.performPolicy(s0, num_episode)
            # print(self.action_t.name)
            step_in_episode = 0
            is_done = False
            while not is_done:
                # add code here
                s1, r1, is_done, info = self.act(a0)
                self.env.render()
                s1 = str(s1)
                self._assert_state_in_QE(s1, randomized = True)
                
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

                if num_episode == max_num_episode:
                    # print current action series
                    print("t:{0:>2}: s:{1}, a:{2:10}, s1:{3}".
                          format(step_in_episode, s0, a0, s1))
                
                s0, a0 = s1, a1
                step_in_episode += 1

            print("Episode {0} takes {1} steps.".format(
                num_episode, step_in_episode))
            total_steps += step_in_episode
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
    '''使用近似的价值函数实现的Q学习个体
    '''
    def __init__(self, env: Env = None,
                       trans_capacity = 20000,
                       hidden_dim: int = 16):
        if env is None:
            raise "agent should have an environment"
        super(ApproxQAgent, self).__init__(env, trans_capacity)
        self.input_dim, self.output_dim = 1, 1
        if isinstance(env.observation_space, spaces.Discrete):
            self.input_dim = 1
        elif isinstance(env.observation_space, spaces.Box):
            self.input_dim = env.observation_space.shape[0]

        if isinstance(env.action_space, spaces.Discrete):
            self.output_dim = env.action_space.n
        elif isinstance(env.action_space, spaces.Box):
            self.output_dim = env.action_space.shape[0]

        # print("{},{}".format(self.input_dim, self.output_dim))
        self.hidden_dim = hidden_dim
        self.Q = Approximator(dim_input = self.input_dim,
                              dim_output = self.output_dim,
                              dim_hidden = self.hidden_dim)
        self.PQ = self.Q.clone() # 更新参数的网络
        return

    def _decayed_epsilon(self,cur_episode: int, 
                              min_epsilon: float, 
                              max_epsilon: float, 
                              target_episode: int) -> float:
        '''获得一个在一定范围内的epsilon
        '''
        slope = (min_epsilon - max_epsilon) / (target_episode)
        intercept = max_epsilon
        return max(min_epsilon, slope * cur_episode + intercept)

    def _curPolicy(self, s, epsilon = None):
        '''依据更新策略的价值函数(网络)产生一个行为
        '''
        Q_s = self.PQ(s)
        rand_value = random()
        if epsilon is not None and rand_value < epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(Q_s))
        
    def performPolicy(self, s, epsilon = None):
        return self._curPolicy(s, epsilon)


    def _update_Q_net(self):
        '''将更新策略的Q网络(连带其参数)复制给输出目标Q值的网络
        '''
        self.Q = self.PQ.clone()

    
    def _learn_from_memory(self, gamma, batch_size, learning_rate, epochs):
        trans_pieces = self.sample(batch_size)  # 随机获取记忆里的Transmition
        states_0 = np.vstack([x.s0 for x in trans_pieces])
        actions_0 = np.array([x.a0 for x in trans_pieces])
        reward_1 = np.array([x.reward for x in trans_pieces])
        is_done = np.array([x.is_done for x in trans_pieces])
        states_1 = np.vstack([x.s1 for x in trans_pieces])

        X_batch = states_0
        y_batch = self.Q(states_0)  # 得到numpy格式的结果

        Q_target = reward_1 + gamma * np.max(self.Q(states_1), axis=1)*\
            (~ is_done) # is_done则Q_target==reward_1
        y_batch[np.arange(len(X_batch)), actions_0] = Q_target
        # loss is a torch Variable with size of 1
        loss = self.PQ.fit(x = X_batch, 
                           y = y_batch, 
                           learning_rate = learning_rate,
                           epochs = epochs)

        mean_loss = loss.sum().data[0] / batch_size
        self._update_Q_net()
        return mean_loss

    def learning(self, gamma = 0.99,
                       learning_rate=1e-5, 
                       max_episodes=1000, 
                       batch_size = 64,
                       min_epsilon = 0.2,
                       epsilon_factor = 0.1,
                       epochs = 1):

        total_steps, step_in_episode, num_episode = 0, 0, 0
        target_episode = max_episodes * epsilon_factor
        while num_episode < max_episodes:
            epsilon = self._decayed_epsilon(cur_episode = num_episode,
                                            min_epsilon = min_epsilon, 
                                            max_epsilon = 1,
                                            target_episode = target_episode)
            self.state = self.env.reset()
            # self.env.render()
            step_in_episode = 0
            loss, mean_loss = 0.00, 0.00
            is_done = False
            while not is_done:
                s0 = self.state

                a0  = self.performPolicy(s0, epsilon)
                s1, r1, is_done, info, total_reward = self.act(a0)
                # self.env.render()
                step_in_episode += 1
                
                if self.total_trans > batch_size:
                    loss += self._learn_from_memory(gamma, 
                                                    batch_size, 
                                                    learning_rate,
                                                    epochs)
            mean_loss = loss / step_in_episode
            print("{0} epsilon:{1:3.2f}, loss:{2:.3f}".
                format(self.experience.last, epsilon, mean_loss))
            # print(self.experience)
            total_steps += step_in_episode
            num_episode += 1

        return   


def testApproxQAgent():
    env = gym.make("PuckWorld-v0")
    #env = SimpleGridWorld()
    directory = "/home/qiang/workspace/reinforce/python/monitor"
    
    env = gym.wrappers.Monitor(env, directory, force=True)
    agent = ApproxQAgent(env,
                         trans_capacity = 50000, 
                         hidden_dim = 32)
    env.reset()
    print("Learning...")  
    agent.learning(gamma=0.99, 
                   learning_rate = 1e-3,
                   batch_size = 64,
                   max_episodes=5000,   # 最大训练Episode数量
                   min_epsilon = 0.2,   # 最小Epsilon
                   epsilon_factor = 0.3,# 开始使用最小Epsilon时Episode的序号占最大
                                        # Episodes序号之比，该比值越小，表示使用
                                        # min_epsilon的episode越多
                    epochs = 2          # 每个batch_size训练的次数
                   )


if __name__ == "__main__":
    testApproxQAgent()