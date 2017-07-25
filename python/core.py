from random import random, choice
import gym
from gym import Env
import numpy as np
from collections import namedtuple
from typing import List

Transition = namedtuple("Transition", field_names=["state",
                                              "action",
                                              "reward",
                                              "is_done",
                                              "next_state"])
class Episode(object):
    def __init__(self) -> None:
        self.total_reward = 0           # 总的获得的奖励
        self.len = 0                 # episode长度
        self.trans_list = []            # 一次状态转移
        self.cur_pos = -1               # 当前位置
        self.name = str(self.len)                # 名称

    def push(self, trans:Transition) -> float:
        self.trans_list.append(trans)
        self.total_reward += trans[2]
        self.len += 1
        return self.total_reward

    def pop(self) -> Transition:
        if self.len > 1:
            self.len -= 1
            return self.trans_list.pop()
        else:
            return None
    
    def is_complete(self) -> bool:
        if self.len <= 0: return None 
        return self.trans_list[self.len-1].is_done

    def sample(self, batch_size):
        return random.sample(self.trans_list, k = batch_size)

    def __len__(self) -> int:
        return self.len


class ReplayMemory(object):
    """Replay Memory
    Attributes:
        capacity (int): Size of the memory
        memory (List[Transition]): Internal memory to store `Transition`s
        position (int): Index to push value
    """

    def __init__(self, capacity: int) -> None:
        """Creates a ReplayMemory with given `capacity`
        Args:
            capacity (int): Max capacity of the memory
        """
        self.capacity = capacity
        self.position = 0
        self.memory = []

    def push(self, s: np.ndarray, 
                   a: np.ndarray, 
                   r: np.ndarray, 
                   d: bool, 
                   s2: np.ndarray) -> None:
        """Stores values to the memory by creating a `Transition`
        Args:
            s (np.ndarray): State, shape (n, input_dim)
            a (np.ndarray): Action, shape (n, output_dim)
            r (np.ndarray): Reward, shape (n, 1)
            d (bool): If `state` is a terminal state
            s2 (np.ndarray): Next state, shape (n, input_dim)
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        s = np.ravel(s)
        s2 = np.ravel(s2)
        a = np.squeeze(a)

        self.memory[self.position] = Transition(s, a, r, d, s2)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        """Returns a mini-batch of the memory
        Args:
            batch_size (int): Size of mini-batch
        Returns:
            List[Transition]: Mini-batch of `Transitions`
        """
        return random.sample(self.memory, k=batch_size)

    def __len__(self) -> int:
        """Returns the current size of the memory
        Returns:
            int: Current size of the memory
        """
        return len(self.memory)

class Agent(object):
    def __init__(self, env: Env = None, 
                       memory_capacity = 0):
        # 保存一些Agent可以观测到的环境信息以及已经学到的经验
        self.env = env
        self.obs_space = env.observation_space if env is not None else None
        self.action_space = env.action_space if env is not None else None
        self.value_fun = None           
        self.policy_fun = None
        self.memory = ReplayMemory(capacity = memory_capacity)
        self.episodes = []
        self.episode = Episode()
        self.state = None
    
    def performPolicy(self, s, policy_fun):
        if policy_fun is None:
            if self.policy_fun is None:
                return self.action_space.sample()
            return self.policy_fun(s)
        return policy_fun(s)
    
    def act(self, a):
        s1, r, is_done, info = self.env.step(a)
        # TODO add extra code here
        trans = Transition(self.state, a, r, is_done, info)
        total_reward = self.episode.push(trans)
        if is_done:
            self.episodes.append(self.episode)
            self.episode = Episode()
        self.state = s1
        return s1, r, is_done, info, total_reward

    def set_policy(self, policy_fun):
        self.policy_fun = policy_fun

    def set_value_fun(self, value_fun):
        self.value_fun = value_fun

    def learn(self):
        raise ValueError("learn method should be rewrite by a subclass")

