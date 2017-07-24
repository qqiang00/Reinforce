"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_position = -1.2        # 位置是水平位置
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5

        # 作为观察空间每一个特征值的下限
        self.low = np.array([self.min_position, -self.max_speed])   
        self.high = np.array([self.max_position, self.max_speed])

        self.viewer = None
        # 0,1,2代表三个不同的动作减速，不加不减，加速；
        self.action_space = spaces.Discrete(3)  
        # 观察空间由low和high决定
        self.observation_space = spaces.Box(self.low, self.high)    

        self._seed()    # 产生一个随机数种子
        self.reset()

    def _seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)  
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state # 获取位置（水平）和速度
        velocity += (action-1)*0.001 + math.cos(3*position)*(-0.0025)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position)
        reward = -1.0

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    def _height(self, xs):
        # sin内部的数字3与加速度计算中cos内的3是相同的，根据坡度的斜率计算加速度。
        return np.sin(3 * xs)*.45+.55

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600      # 先确定屏幕尺寸
        screen_height = 400 

        world_width = self.max_position - self.min_position # 确定世界尺寸
        scale = screen_width/world_width    # 计算两者映射关系
        carwidth=40             # 随后都是用世界尺寸来描述
        carheight=20

        # 如果还没有设定屏幕对象，则初始化整个屏幕具备的元素。
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # 在Viewer里绘制一个几何图像的步骤如下：
            # 1. 建立该对象需要的数据本身
            # 2. 使用rendering提供的方法返回一个geom对象
            # 3. 对geom对象进行一些对象颜色、线宽、线型、变换属性的设置（有些对象提供一些个性化的方法
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

            # 绘制上下坡，使用100个点的连线来模拟
            xs = np.linspace(self.min_position, self.max_position, 100)
            # ys与xs的关系函数
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            # 调用rendering内部的提供的常用绘制函数。
            self.track = rendering.make_polyline(xys)
            self.track.set_color(0.1, 0.9, 0.1)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)
            
            # 小车长方形车身与坡之间的间隙
            clearance = 10
            # 下面绘制的起点在(0,0)处
            # 画小车,小车的左右上下边界
            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            # car 是一个geom对象
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            # 将车身位置做个变换(0，clearance)，上移clearance长度
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            
            # 得到一个小车的转换对象，为今后改变小车位置和方向做准备
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            # 把小车添加至geom中
            self.viewer.add_geom(car)
            
            # 前轮对象make_circle提供三个参数(半径,绘制解析率,是否填充),均有默认值
            # 解析率为30: 将圆周划分为30等分进行绘制，默认为填充
            frontwheel = rendering.make_circle(carheight/2.5, 30, False)
            frontwheel.set_color(.9, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,
                                                                clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,
                                                                clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10)
                , (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        # 如果已经为屏幕准备好了要绘制的对象
        # 本例中唯一要做的就是改变小车的位置和旋转
        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, 
            self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


if __name__ =="__main__":
    env = MountainCarEnv()
    print("hello")
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())

    print("env closed")