import random    # 随机策略时用到
import matplotlib.pyplot as plt
import numpy as np


def learning_curve(data, x_index = 0, y1_index = 1, y2_index = None, title = "", 
                   x_name = "", y_name = "",
                   y1_legend = "", y2_legend=""):
    '''根据统计数据绘制学习曲线，
    Args:
        statistics: 数据元组，每一个元素是一个列表，各列表长度一致 ([], [], [])
        x_index: x轴使用的数据list在元组tuple中索引值
        y_index: y轴使用的数据list在元组tuple中的索引值
        title:图标的标题
        x_name: x轴的名称
        y_name: y轴的名称
        y1_legend: y1图例
        y2_legend: y2图例
    Return:
        None 绘制曲线图
    '''
    fig, ax = plt.subplots()
    x = data[x_index]
    y1 = data[y1_index]
    ax.plot(x, y1, label = y1_legend)
    if y2_index is not None:
        ax.plot(x, data[y2_index], label = y2_legend)
    ax.grid(True, linestyle='-.')
    ax.tick_params(labelcolor='black', labelsize='medium', width=1)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    ax.legend()
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    #plt.axis([40, 160, 0, 0.03])
    #plt.grid(True)
    plt.show()    
    
def str_key(*args):
    '''将参数用"_"连接起来作为字典的键，需注意参数本身可能会是tuple或者list型，
    比如类似((a,b,c),d)的形式。
    '''
    new_arg = []
    for arg in args:
        if type(arg) in [tuple, list]:
            new_arg += [str(i) for i in arg]
        else:
            if arg is None:
                pass
            else:
                new_arg.append(str(arg))
    return "_".join(new_arg)

def set_dict(target_dict, value, *args):
    if target_dict is None:
        return
    target_dict[str_key(*args)] = value

def get_dict(target_dict, *args):
    #print("key: {}".format(str_key(*args)))
    if target_dict is None:
        return
    return target_dict.get(str_key(*args),0)


def uniform_random_pi(A, s = None, Q = None, a = None):
    '''均一随机策略下某行为的概率
    '''
    n = len(A)
    if n == 0:
        return 0.0
    return 1.0/n

def sample(A):
    '''从A中随机选一个
    '''
    return random.choice(A) # 随机选择A中的一个元素

def uniform_random_policy(A, s = None, Q = None):
    '''均一随机策略
    '''
    return sample(A)


def greedy_pi(A, s, Q, a):
    '''依据贪婪选择，计算在行为空间A中，状态s下，a行为被贪婪选中的几率
    考虑多个行为的价值相同的情况
    '''
    #print("in greedy_pi: s={},a={}".format(s,a))
    max_q, a_max_q = -float('inf'), []
    for a_opt in A:# 统计后续状态的最大价值以及到达到达该状态的行为（可能不止一个）
        q = get_dict(Q, s, a_opt)
        #print("get q from dict Q:{}".format(q))
        if q > max_q:
            max_q = q
            a_max_q = [a_opt]
        elif q == max_q:
            #print("in greedy_pi: {} == {}".format(q,max_q))
            a_max_q.append(a_opt)
    n = len(a_max_q)
    if n == 0: return 0.0
    return 1.0/n if a in a_max_q else 0.0


def greedy_policy(A, s, Q, epsilon = None):
    """在给定一个状态下，从行为空间A中选择一个行为a，使得Q(s,a) = max(Q(s,))
    考虑到多个行为价值相同的情况
    """
    max_q, a_max_q = -float('inf'), []
    for a_opt in A:
        q = get_dict(Q, s, a_opt)
        if q > max_q:
            max_q = q
            a_max_q = [a_opt]
        elif q == max_q:
            a_max_q.append(a_opt)
    return random.choice(a_max_q)

        
def epsilon_greedy_pi(A, s, Q, a, epsilon = 0.1):
    m = len(A)
    if m == 0: return 0.0
    greedy_p = greedy_pi(A, s, Q, a)
    #print("greedy prob:{}".format(greedy_p))
    if greedy_p == 0:
        return epsilon / m
    n = int(1.0/greedy_p)
    return (1 - epsilon) * greedy_p + epsilon/m


def epsilon_greedy_policy(A, s, Q, epsilon = 0.05):
    rand_value = random.random()
    if rand_value < epsilon:
        return sample(A)
    else:
        return greedy_policy(A, s, Q)