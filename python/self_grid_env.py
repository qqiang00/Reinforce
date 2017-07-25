"""
This file illustrates how to personalize your gridworld environment
by using GridWorldEnv

License: MIT

Author: Qiang Ye
Date: July 25, 2017
"""
from gridworld import GridWorldEnv
from gym import spaces

from gridworld import GridWorldEnv
env = GridWorldEnv(n_width=12,          # 水平方向格子数量
                   n_height = 4,        # 垂直方向格子数量
                   u_size = 60,         # 可以根据喜好调整大小
                   default_reward = -1, # 默认格子的即时奖励值
                   default_type = 0)    # 默认的格子都是可以进入的
env.action_space = spaces.Discrete(4)   # 设置行为空间数量
# 格子世界环境类默认使用0表示左，1：右，2：上，3:下，4,5,6,7为斜向行走
# 具体可参考_step内的定义
# 格子世界的观测空间不需要额外设置，会自动根据传输的格子数量计算得到

env.start = (0,0)
env.ends = [(11,0)]

for i in range(10):
    env.rewards.append((i+1,0,-100))
    env.ends.append((i+1,0))

env.types = [(5,1,1),(5,2,1)]

env.refresh_setting()
env.reset()
env.render()
input("press any key to continue...")