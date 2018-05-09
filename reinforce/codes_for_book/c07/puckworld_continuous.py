"""
PuckWorld Environment for OpenAI gym

The data used in this model comes from:
http://cs.stanford.edu/people/karpathy/reinforcejs/puckworld.html


Author: Qiang Ye
Date: July 17, 2017
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

RAD2DEG = 57.29577951308232     # 弧度与角度换算关系1弧度=57.29..角度

class PuckWorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
        }

    def __init__(self):
        self.width = 400            # 场景宽度 screen width
        self.height = 400           # 场景长度
        self.l_unit = 1.0           # 场景长度单位 pysical world width
        self.v_unit = 1.0           # 速度单位 velocity 
        self.max_speed = 0.025      # max agent velocity along a axis
        
        self.min_action = -1.0      # 最小行为值
        self.max_action = 1.0       # 最大行为值
        self.re_pos_interval = 30   # 目标重置距离时间
        self.accel = 0.002          # agent 加速度
        self.rad = 0.05             # agent 半径,目标半径
        self.target_rad = 0.01      # target radius.
        self.goal_dis = self.rad    # 目标接近距离 expected goal distance
        self.t = 0                  # puck world clock
        self.update_time = 100      # time for target randomize its position
        # 作为观察空间每一个特征值的下限
        self.low_state = np.array([0,        # agent position x
                                   0,
                                   -self.max_speed,    # agent velocity
                                   -self.max_speed,    
                                   0,         # target position x
                                   0,
                                   ])   
        self.high_state = np.array([self.l_unit,                 
                                   self.l_unit,
                                   self.max_speed,    
                                   self.max_speed,    
                                   self.l_unit,    
                                   self.l_unit,
                                   ])   
        self.reward = 0         # for rendering
        self.action = None      # for rendering
        self.viewer = None
        # 0,1,2,3,4 represent left, right, up, down, -, five moves.
        self.action_space = spaces.Box(low = self.min_action, high = self.max_action, shape=(2,), dtype = np.float32)
        # 观察空间由low和high决定
        self.observation_space = spaces.Box(self.low_state, self.high_state, dtype = np.float32)    

        self.seed()    # 产生一个随机数种子
        self.reset()

    def seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)  
        return [seed]

    def _clip(self, x, min, max):
        if x < min:
            return min
        elif x > max:
            return max
        return x
    
    def step(self, action):
        self.action = action    # action for rendering
        ppx,ppy,pvx,pvy,tx,ty = self.state # 获取agent位置，速度，目标位置
        ppx, ppy = ppx+pvx, ppy+pvy         # update agent position
        pvx, pvy = pvx*0.95, pvy*0.95       # natural velocity loss

        pvx += self.accel * action[0]   # right
        pvy += self.accel * action[1]   # up
        
        pvx = self._clip(pvx, -self.max_speed, self.max_speed)
        pvy = self._clip(pvy, -self.max_speed, self.max_speed)
        
        if ppx < self.rad:              # encounter left bound
            #pvx *= -0.5
            ppx = self.rad
        if ppx > 1 - self.rad:          # right bound
            #pvx *= -0.5
            ppx = 1 - self.rad
        if ppy < self.rad:              # bottom bound
            #pvy *= -0.5
            ppy = self.rad
        if ppy > 1 - self.rad:          # right bound
            #pvy *= -0.5
            ppy = 1 - self.rad
        
        self.t += 1
        if self.t % self.update_time == 0:  # update target position
            tx = self._random_pos()         # randomly
            ty = self._random_pos()

        dx, dy = ppx - tx, ppy - ty         # calculate distance from
        dis = self._compute_dis(dx, dy)     # agent to target

        self.reward = self.goal_dis - dis   # give an reward

        done = bool(dis <= self.goal_dis)   
        
        self.state = (ppx, ppy, pvx, pvy, tx, ty)
        return np.array(self.state), self.reward, done, {}

    def _random_pos(self):
        return self.np_random.uniform(low = 0, high = self.l_unit)

    def _compute_dis(self, dx, dy):
        return math.sqrt(math.pow(dx,2) + math.pow(dy,2))

    def reset(self):
        self.state = np.array([ self._random_pos(),
                                self._random_pos(),
                                0,
                                0,
                                self._random_pos(),
                                self._random_pos()
                               ])
        return self.state   # np.array(self.state)


    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        scale = self.width/self.l_unit      # 计算两者映射关系
        rad = self.rad * scale              # 随后都是用世界尺寸来描述
        t_rad = self.target_rad * scale     # target radius
        
        
        action = self.action
        if action is None:
            length = 0.00
        else:
            length = np.sqrt(np.sum(np.dot(action, action)))
        #print(action, length)
        # 如果还没有设定屏幕对象，则初始化整个屏幕具备的元素。
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width, self.height)

            # 在Viewer里绘制一个几何图像的步骤如下：
            # 1. 建立该对象需要的数据本身
            # 2. 使用rendering提供的方法返回一个geom对象
            # 3. 对geom对象进行一些对象颜色、线宽、线型、变换属性的设置（有些对象提供一些个
            #    性化的方法
            #    来设置属性，具体请参考继承自这些Geom的对象），这其中有一个重要的属性就是
            #    变换属性，该属性负责对对象在屏幕中的位置、渲染、缩放进行渲染。如果某对象
            #    在呈现时可能发生上述变化，则应建立关于该对象的变换属性。该属性是一个
            #    Transform对象，而一个Transform对象，包括translate、rotate和scale
            #    三个属性，每个属性都由以np.array对象描述的矩阵决定。
            # 4. 将新建立的geom对象添加至viewer的绘制对象列表里，如果在屏幕上只出现一次，
            #    将其加入到add_onegeom(）列表中，如果需要多次渲染，则将其加入add_geom()
            # 5. 在渲染整个viewer之前，对有需要的geom的参数进行修改，修改主要基于该对象
            #    的Transform对象
            # 6. 调用Viewer的render()方法进行绘制

            target = rendering.make_circle(t_rad, 30, True)
            target.set_color(0.1, 0.9, 0.1)
            self.viewer.add_geom(target)
            target_circle = rendering.make_circle(t_rad, 30, False)
            target_circle.set_color(0, 0, 0)
            self.viewer.add_geom(target_circle)
            self.target_trans = rendering.Transform()
            target.add_attr(self.target_trans)
            target_circle.add_attr(self.target_trans)

            self.agent = rendering.make_circle(rad, 30, True)
            self.agent.set_color(0, 1, 0)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)

            agent_circle = rendering.make_circle(rad, 30, False)
            agent_circle.set_color(0, 0, 0)
            agent_circle.add_attr(self.agent_trans)
            self.viewer.add_geom(agent_circle)

            start_p = (0, 0)
            end_p = (0.7 * rad * length, 0)
            self.line = rendering.Line(start_p, end_p)
            self.line.linewidth = rad / 10
            
            self.line_trans = rendering.Transform()
            self.line.add_attr(self.line_trans)
            self.viewer.add_geom(self.line)
            
            self.arrow = rendering.FilledPolygon([
                (0.7*rad*length,0.15*rad),
                (rad*length,0),
                (0.7*rad*length,-0.15*rad)
                ])
            self.arrow.set_color(0,0,0)
            self.arrow.add_attr(self.line_trans)
            self.viewer.add_geom(self.arrow)
            

        # 如果已经为屏幕准备好了要绘制的对象
        # 本例中唯一要做的就是改变小车的位置和旋转
        ppx,ppy,_,_,tx,ty = self.state
        self.target_trans.set_translation(tx*scale, ty*scale)
        self.agent_trans.set_translation(ppx*scale, ppy*scale)
        # 按距离给Agent着色
        vv, ms = self.reward + 0.3, 1
        r, g, b, = 0, 1, 0
        if vv >= 0:
            r, g, b = 1 - ms*vv, 1, 1 - ms*vv
        else:
            r, g, b = 1, 1 + ms*vv, 1 + ms*vv 
        self.agent.set_color(r, g, b)
        

        if length == 0:
            self.line.set_color(r,g,b)
            self.arrow.set_color(r,g,b) # 背景色
        else:
            if action[1] >= 0:
                rotate = math.acos(action[0]/length) # action[0]水平方向
            else:# 垂直方向
                rotate = 2 * math.pi - math.acos(action[0]/length)
            self.line_trans.set_translation(ppx*scale, ppy*scale)
            self.line_trans.set_rotation(rotate)
            self.line.set_color(0,0,0)
            self.arrow.set_color(0,0,0)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    
    def close(self):
        if self.viewer: self.viewer.close()

if __name__ =="__main__":
    
    import time
    
    env = PuckWorldEnv()
    print("hello")
    env.reset()
    nfs = env.observation_space.shape[0]
    nfa = env.action_space.shape[0]
    print("nfs:{}; nfa:{}".format(nfs,nfa))
    print(env.observation_space)
    print(env.action_space)

    for _ in range(10000):
        action = env.action_space.sample()
        env.step(action)
        env.render()
        time.sleep(1)
    print("env closed")