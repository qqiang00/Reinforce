class Environment():
    '''环境类。环境需要描述这个世界，响应与其进行交互的个体的行为，并向个体提供交互比较的信息
    '''
    def __init__(self):
        # 个体在环境中所具有的行为和状态空间
        self.action_space = None
        self.status_space = None
        print("构建环境完毕")
        pass


    def dynamics(self, action):
        obs_1, reward_1, is_done = None, None, False
        # 确定个体的观测状态、即时奖励、是否交互结束
        pass
        # 返回给个体的信息 
        print("环境: 获得个体行为，处理后返回给个体观测和及时奖励，并告知个体是否交互结束")
        return obs_1, reward_1, is_done
    

    def obs_space(self):
        '''
        根据自身的状态空间以及交互的个体身份确定该个体所具有的观测空间
        '''
        # 可默认具有环境的状态空间，此时对该个体来说是完全可观测环境
        print("环境:开放观测空间给个体")
        obs_space = self.status_space 
        return obs_space
        

    def act_space(self):
        '''根据环境的行为空间以及交互的个体身份确定该个体所具有的行为空间
        '''
        print("环境:开放行为空间给个体")
        act_space = self.action_space
        return act_space
        

    def reset(self):
        '''重新设定环境信息，给个体一个初始观测
        '''
        print("重置环境信息")
        agent_start_obs = None # 初始状态下个体的观测
        return agent_start_obs
    


class Agent():
    '''个体类。
    '''
    def __init__(self, env = None, name = "agent1"):
        self.env = env
        self.name = name
        self.act_space = env.act_space()
        self.obs_space = env.obs_space()

        self.values = None
        self.policy = None
        self.memory = None

        self.obs_0 = None # t=0时刻个体的观测
        print("构建个体完毕")
        pass


    def __str__(self):
        return self.name
    

    def update_values(self):
        print("个体：更新观测状态价值")
        # self.values = # Code here
        pass
    

    def update_policy(self):
        print("个体：更新策略")
        # self.policy = # Code here
        pass


    def update_model(self):
        print("个体：更新模型")
        # self.model = # Code here
        pass 


    def update_memory(self,
                      obs_0 = None,
                      action_0 = None,
                      reward_1 = None,
                      is_done = None,
                      obs_1 = None):
        print("个体：当前状态转换加入记忆中")
        # self.memory = # Code here
        pass


    def perform_policy(self, policy = None, obs = None):
        # 产生一个行为
        if policy is not None:
            action = policy(obs)
        else:
            action = None # 随即产生
        print("个体：依据策略产生一个行为")
        return action


    def model(self, action = None):
        print("个体：思考行为可能带来的下一时刻的观测、及时奖励及是否交互结束")
        # 思考个体的观测状态、即时奖励、是否交互结束
        v_obs_1, v_reward_1, v_is_done = None, None, None
        # 依据action确定v_obs_1, v_reward_1, v_is_done
        # 返回给个体的虚拟信息 
        # 也可以把思考的过程变为记忆的一部分
        return v_obs_1, v_reward_1, v_is_done
        

    def act(self, action_0):
        # 调用环境的动力学方法
        print("个体：执行一个行为")
        obs_1, reward_1, is_done = self.env.dynamics(action_0)
        self.update_memory(self.obs_0, action_0, reward_1, is_done, obs_1)
        self.obs_0 = obs_1
        pass
    
    
    def learning(self):
        '''个体的学习过程
        '''
        self.obs_0 = env.reset()
        policy = None # 选定一个策略
        end_condition = False # 设定一个终止条件
        while( not end_condition):
            obs = self.obs_0
            act_0 = self.perform_policy(policy, obs)
            self.act(act_0)
            self.update_policy()
            # addtional code here
            end_condition = True
        pass

    def planning(self):
        '''个体的规划过程
        '''
        policy = None # 选定一个策略
        end_condition = False # 设定一个终止条件
        obs_0 = None # 选定一个观测状态
        while(not end_condition):
            obs = self.obs_0
            v_act_0 = self.perform_policy(policy, obs)
            self.model(action = v_act_0)
            self.update_policy()  
            # addtional code here
            end_condition = True
        pass

if __name__ == "__main__":
    env = Environment()
    agent = Agent(env = env, name = "agent_1")
    env.reset()
    act = agent.perform_policy(None, agent.obs_0)
    agent.act(act)
    agent.learning()
    agent.planning()