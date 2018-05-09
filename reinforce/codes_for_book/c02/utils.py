# 辅助函数
def str_key(*args):
    '''将参数用"_"连接起来作为字典的键，需注意参数本身可能会是tuple或者list型，
    比如类似((a,b,c),d)的形式。
    '''
    new_arg = []
    for arg in args:
        if type(arg) in [tuple, list]:
            new_arg += [str(i) for i in arg]
        else:
            new_arg.append(str(arg))
    return "_".join(new_arg)

def set_dict(target_dict, value, *args):
    target_dict[str_key(*args)] = value
    
def set_prob(P, s, a, s1, p = 1.0): # 设置概率字典
    set_dict(P, p, s, a, s1)

def get_prob(P, s, a, s1): # 获取概率值
    return P.get(str_key(s,a,s1), 0)
    
def set_reward(R, s, a, r): # 设置奖励字典
    set_dict(R, r, s, a)

def get_reward(R, s, a): # 获取奖励值
    return R.get(str_key(s,a), 0)

def display_dict(target_dict): # 显示字典内容
    for key in target_dict.keys():
        print("{}:　{:.2f}".format(key, target_dict[key]))
    print("")
    
# 辅助方法
def set_value(V, s, v): # 设置价值字典
    set_dict(V, v, s)
    
def get_value(V, s): # 获取价值值
    return V.get(str_key(s), 0)

def set_pi(Pi, s, a, p = 0.5): # 设置策略字典
    set_dict(Pi, p, s, a)
    
def get_pi(Pi, s, a): # 获取策略（概率）值
    return Pi.get(str_key(s,a), 0)


