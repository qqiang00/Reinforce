#!/home/qiang/PythonEnv/venv/bin/python3.5
# -*- coding: utf-8 -*-
# #有风格子世界

from random import random, choice


class Agent():
    def __init__(self):
        # 保存一些Agent可以观测到的环境信息以及已经学到的经验
        self.Q = {}  # {Position_name:{action:q_value}}
        self.E = {}
        # Agent knows what are the actions he could take
        self.actions = {}
        self.policy = None
        return

    def loadActions(self, env):
        env.loadActions(self)
        return

    def initValueOfPositionName(self, Position_name, randomized=True):
        if not self.__isPositionNameInQE(Position_name):
            self.Q[Position_name] = {}
            self.E[Position_name] = {}
            for key in self.actions.keys():
                default_value = random() / 10 if randomized is True else 0.0
                self.Q[Position_name][key] = default_value
                self.E[Position_name][key] = 0

    # update the info that the agent could observe from environment
    def observe(self, env):
        # the return of the env.releaseInfoTo(agent) function is the
        # new Position in which the agent should be
        # agent's reward_t1 is also updated in that function.
        return env.releaseInfo()

    # take an action
    def performAction(self, action, env):
        env.model(action)

    # Agent依据当前策略和状态决定下一步的动作
    def performPolicy(self, Position_name, episode_num):
        return self.__curPolicy(Position_name, episode_num)

    # sarsa learning

    def sarsaLearning(self, env, gamma, alpha, max_episode_num):
        env.loadActions(self)
        self.__initValue(env)
        # self.Position_t_name, self.reward_t1 = self.observe(env)
        total_time = 0
        time_in_episode = 0
        num_episode = 1
        while num_episode <= max_episode_num:
            env.reset()
            Position_t_name, _ = self.observe(env)
            action_t = self.performPolicy(Position_t_name, num_episode)
            # print(self.action_t.name)
            time_in_episode = 0
            while not self.__isNowInGoalPosition(env):
                # add code here
                self.performAction(action_t, env)
                Position_t1_name, reward_t1 = self.observe(env)

                if not self.__isPositionNameInQE(Position_t1_name):
                    self.initValueOfPositionName(
                        Position_t1_name, randomized=True)

                action_t1 = self.performPolicy(Position_t1_name, num_episode)
                old_q = self.__getQValueOf(Position_t_name, action_t.name)
                q_prime = self.__getQValueOf(Position_t1_name, action_t1.name)
                td_target = reward_t1 + gamma * q_prime
                #alpha = alpha / num_episode
                new_q = old_q + alpha * (td_target - old_q)
                self.__setQValueOf(Position_t_name, action_t.name, new_q)

                if num_episode == max_episode_num:
                    # print current action series
                    print("T:{0:<2}: S:{1}, A:{2:10}, S1:{3}".
                          format(time_in_episode, Position_t_name,
                                 action_t.name, Position_t1_name))
                Position_t_name = Position_t1_name
                action_t = action_t1
                time_in_episode += 1
            print("Episode {0} takes time {1}".
                  format(num_episode, time_in_episode))
            total_time += time_in_episode
            num_episode += 1
        return

    def sarsaLambdaLearning(self, env, lambda_, gamma, alpha, max_episode_num):
        env.loadActions(self)
        self.__initValue(env)
        total_time = 0
        time_in_episode = 0
        num_episode = 1
        while num_episode <= max_episode_num:
            self.__resetEValue()
            env.reset()
            Position_t_name, _ = self.observe(env)
            action_t = self.performPolicy(Position_t_name, num_episode)
            # print(self.action_t.name)
            time_in_episode = 0
            while not self.__isNowInGoalPosition(env):
                # add code here
                self.performAction(action_t, env)
                Position_t1_name, reward_t1 = self.observe(env)

                if not self.__isPositionNameInQE(Position_t1_name):
                    self.initValueOfPositionName(
                        Position_t1_name, randomized=True)

                action_t1 = self.performPolicy(Position_t1_name, num_episode)

                q = self.__getQValueOf(Position_t_name, action_t.name)
                q_prime = self.__getQValueOf(Position_t1_name, action_t1.name)
                delta = reward_t1 + gamma * q_prime - q

                e = self.__getEValueOf(Position_t_name, action_t.name)
                e = e + 1
                self.__setEValueOf(Position_t_name, action_t.name, e)

                Positions_actions_list = list(zip(self.E.keys(),
                                                  self.E.values()))
                for Position_name, actions_es_dic in Positions_actions_list:
                    actions_es_list = list(zip(actions_es_dic.keys(),
                                               actions_es_dic.values()))
                    for action_name, e_value in actions_es_list:

                        old_q = self.__getQValueOf(Position_name, action_name)
                        #alpha = alpha / num_episode
                        new_q = old_q + alpha * delta * e_value
                        new_e = gamma * lambda_ * e_value
                        self.__setQValueOf(Position_name, action_name, new_q)
                        self.__setEValueOf(Position_name, action_name, new_e)

                if num_episode == max_episode_num:
                    # print current action series
                    print("T:{0:<2}: S:{1}, A:{2:10}, S1:{3}".
                          format(time_in_episode, Position_t_name,
                                 action_t.name, Position_t1_name))
                Position_t_name = Position_t1_name
                action_t = action_t1
                time_in_episode += 1

            print("Episode {0} takes time {1}".format(
                num_episode, time_in_episode))
            total_time += time_in_episode
            num_episode += 1
        return
    # do initialization of QValue

    def __initValue(self, env):
        env.initValue(self)

    # return a possible action list for a given Position
    # def possibleActionsForPosition(self, Position):
    #  actions = []
    #  # add your code here
    #  return actions

    # if a Position exists in Q dictionary
    def __isPositionNameInQE(self, Position_name):
        return self.Q.get(Position_name) is not None

    # return an action
    def __curPolicy(self, Position_name, episode_num):
        epsilon = 1.00 / episode_num
        QS = self.Q[Position_name]
        action_str_with_max_Q = "act_str"
        rand_value = random()
        action = None
        if rand_value > epsilon:  # select actions that have max q
            action_str_with_max_Q = max(QS, key=QS.get)
            action = self.actions[action_str_with_max_Q]
        else:
            action = choice(list(self.actions.values()))

        # print("{0}: {1},{2} Q:{3}".format(Position.name, action_str_with_max_Q, action.name, QS))
        return action

    # ensure that QValue for Position is available; otherwise initialize it
    # with a argument randomized, if randomized is false, use 0 to initialize
    # all Qvalue of current Position, else use random number between (0,1)
    def __assertValueExistsForPositionName(self, Position_name, randomized=True):
        # 　cann't find the Position
        if not self.__isPositionNameInQE(Position_name):
            self.initValueOfPositionName(Position_name, randomized)

    def __getQValueOf(self, Position_name, action_name):
        self.__assertValueExistsForPositionName(Position_name, randomized=True)
        return self.Q[Position_name][action_name]

    def __setQValueOf(self, Position_name, action_name, value):
        self.__assertValueExistsForPositionName(Position_name, randomized=True)
        self.Q[Position_name][action_name] = value

    def __getEValueOf(self, Position_name, action_name):
        self.__assertValueExistsForPositionName(Position_name, randomized=True)
        return self.E[Position_name][action_name]

    def __setEValueOf(self, Position_name, action_name, value):
        self.__assertValueExistsForPositionName(Position_name, randomized=True)
        self.E[Position_name][action_name] = value

    def __resetEValue(self):
        for value_dic in self.E.values():
            for key in value_dic.keys():
                value_dic[key] = 0.00

    # agent itself doesn't know which Position is the final Position of an episode
    # it should ask for the environment
    def __isNowInGoalPosition(self, env):
        return env.isAgentInGoalPosition()


class Environment():
    def __init__(self):

        self.startPosition = Position([0, 3])
        self.goalPosition = Position([7, 3])

        self.agentPosition = Position(self.startPosition.pos)
        self.agent_reward_t1 = 0
        self.action_dpos_dict = {
            "left": (-1, 0),
            "right": (1, 0),
            "up": (0, 1),
            "down": (0, -1)  # ,
            #"left_up":(-1,1),
            #"up_right":(1,1),
            #"right_down":(1,-1),
            #"down_left":(-1,-1)
        }
        return

    def loadActions(self, agent):
        agent.actions = {}
        for action_name in self.action_dpos_dict.keys():
            agent.actions[action_name] = Action(name=action_name)

    def reset(self):
        self.agentPosition = Position(self.startPosition.pos)
        return

    # only environment knows the start and goal Position, so it is
    # appropriate that QValue of the two Positions is set in this class
    def initValue(self, agent):
        agent.initValueOfPositionName(self.startPosition.name, randomized=True)
        agent.initValueOfPositionName(self.goalPosition.name, randomized=False)
        return

    def model(self, action):
        old_x, old_y = self.agentPosition.pos[0], self.agentPosition.pos[1]
        new_x, new_y = old_x, old_y

        # windy effect
        if new_x in [3, 4, 5, 8]:
            new_y += 1
        elif new_x in [6, 7]:
            new_y += 2
        # action
        dx, dy = self.action_dpos_dict[action.name]
        new_x += dx
        new_y += dy

        # boundary restriction
        if new_x < 0:
            new_x = 0
        elif new_x >= 9:
            new_x = 9
        if new_y < 0:
            new_y = 0
        elif new_y >= 6:
            new_y = 6

        self.agentPosition = Position([new_x, new_y])
        # reward
        if self.isAgentInGoalPosition():
            self.agent_reward_t1 = 0
        else:
            self.agent_reward_t1 = -1
        # 环境接受状态和动作给出奖励，并更新自己相关的模型参数
        # update self.agentPosition to a new Position according to model
        return

    # notify the reward given to agent set new agent Position as return
    # let agent itself to decide when to update its latest Position

    def releaseInfo(self):
        #agent.reward_t1 = self.agent_reward_t1
        #agent.Position_t_name = self.agentPosition.name
        return self.agentPosition.name, self.agent_reward_t1

    # check is a given Position is a goal Position
    def isAgentInGoalPosition(self):
        return self.agentPosition.equalTo(self.goalPosition)

# Status


class Position():
    def __init__(self, pos=[0, 3]):
        self.pos = pos.copy()  # left bottom corner is (0,0)
        self.name = "X{0}-Y{1}".format(self.pos[0], self.pos[1])
        return

    def rename(self, newName):
        self.name = newName
        return

    def equalTo(self, Position):
        return self.pos[0] == Position.pos[0] and self.pos[1] == Position.pos[1]

# Action


class Action():

    def __init__(self, name="None"):
        self.name = name
        return

    def rename(self, newName):
        self.name = newName
        return


def sarsaLearningExample(agent, env):
    agent.sarsaLearning(env=env,
                        gamma=0.9,
                        alpha=0.1,
                        max_episode_num=1000)


def sarsaLambdaLearningExample(agent, env):
    agent.sarsaLambdaLearning(env=env,
                              lambda_=0.1,
                              gamma=0.9,
                              alpha=0.1,
                              max_episode_num=1000)


if __name__ == "__main__":
    myAgent = Agent()
    windy_grid_env = Environment()
    print("Learning...")
    sarsaLearningExample(myAgent, windy_grid_env)
    #sarsaLambdaLearningExample(myAgent, windy_grid_env)
