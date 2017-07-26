from random import random, choice
import gym
from gym import Env
import numpy as np
from collections import namedtuple
from typing import List
import random
'''
Transition = namedtuple("Transition", field_names=["state",
                                              "action",
                                              "reward",
                                              "is_done",
                                              "next_state"])
'''

class State(object):
    def __init__(self,name):
        self.name = name

  
class Transition(object):
    def __init__(self, s0, a0, reward:float, is_done:bool, s1):
        self.s0 = s0
        self.a0 = a0
        self.reward = reward
        self.is_done = is_done
        self.s1 = s1

    def __str__(self):
        return "s:{0:<3} a:{1:<3} r:{2:<4} is_end:{3:<5} s1:{4:<3}".\
            format(self.s0, self.a0, self.reward, self.is_done, self.s1)


class Episode(object):
    def __init__(self, id:int = 0) -> None:
        self.total_reward = 0       # 总的获得的奖励
        # self.len = 0                # episode长度
        self.trans_list = []        # 状态转移列表
        self.name = str(id)   # 可以给Episode起个名字："成功闯关,黯然失败？"

    def push(self, trans:Transition) -> float:
        self.trans_list.append(trans)
        self.total_reward += trans.reward
        return self.total_reward

    @property
    def len(self):
        return len(self.trans_list)

    def __str__(self):
        return "episode{0:<3} lenth:{1:<4} total_reward:{2:<3}".\
            format(self.name, self.len,self.total_reward)

    def print_detail(self):
        print("detail of ({0}):".format(self))
        for i,trans in enumerate(self.trans_list):
            print("step{0:<3} ".format(i),end=" ")
            print(trans)

    '''
    def pop(self) -> Transition:
        if self.len > 1:
            self.len -= 1
            trans = self.trans_list.pop()
            self.total_reward -= trans[2]
            return trans
        else:
            return None
    '''

    def is_complete(self) -> bool:
        if self.len == 0: return False 
        return self.trans_list[self.len-1].is_done

    def sample(self,batch_size = 1):   
        '''随即产生一个trans
        '''
        return random.sample(self.trans_list, k = batch_size)

    def __len__(self) -> int:
        return self.len

class Experience(object):
    def __init__(self, capacity:int = 20000):
        self.capacity = capacity    # 容量：指的是trans总数量
        self.episodes = []      # episode列表
        self.next_id = 0            # 下一个episode的Id
        self.total_trans = 0    # 总的转换数量
        
    def __str__(self):
        return "exp info:{0} episodes, memory usage {1}/{2}".\
                format(self.len, self.total_trans, self.capacity)

    @property
    def len(self):
        return len(self.episodes)

    def _remove(self,index = 0):      # 扔掉一个Episode，默认第一个
        if index > self.len - 1:
            raise(Exception("invalid index"))
        if self.len > 0:
            episode = self.episodes[index]
            self.episodes.remove(episode)
            # self.len -= 1
            self.total_trans -= episode.len
            return episode
        else:
            return None

    def _remove_first(self):
        self._remove(index = 0)

    def push(self, trans): 
        '''压入一个状态转换
        '''
        if self.capacity <= 0:
            return
        while self.total_trans >= self.capacity: # 可能会有空episode吗？
            episode = self._remove_first()
        cur_episode = None
        if self.len == 0 or self.episodes[self.len-1].is_complete():
            cur_episode = Episode(self.next_id)
            self.next_id += 1
            self.episodes.append(cur_episode)
        else:
            cur_episode = self.episodes[self.len-1]
        self.total_trans += 1
        return cur_episode.push(trans)      #return  total reward of an episode

    def sample(self, batch_size=1): # sample transition
        sample_trans = []
        for _ in range(batch_size):
            index = int(random.random() * self.len)
            sample_trans += self.episodes[index].sample()
        return sample_trans

    def sample_episode(self, episode_num = 1):  # sample episode
        return random.sample(self.episodes, k = episode_num)

    @property
    def last(self):
        if self.len > 0:
            return self.episodes[self.len-1]
        return None

class Agent(object):
    def __init__(self, env: Env = None, 
                       trans_capacity = 0):
        # 保存一些Agent可以观测到的环境信息以及已经学到的经验
        self.env = env
        self.obs_space = env.observation_space if env is not None else None
        self.action_space = env.action_space if env is not None else None
        self.experience = Experience(capacity = trans_capacity)
        self.state = None   # current state of agent
    
    def performPolicy(self,policy_fun, s):
        if policy_fun is None:
            return self.action_space.sample()
        return policy_fun(s)
    
    def act(self, a0):
        s1, r1, is_done, info = self.env.step(a0)
        # TODO add extra code here
        trans = Transition(self.state, a0, r1, is_done, s1)
        total_reward = self.experience.push(trans)
        self.state = s1
        return s1, r1, is_done, info, total_reward

    def learn(self):
        raise NotImplementedError

