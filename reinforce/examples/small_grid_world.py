'''
Implementation of small grid world example illustrated by David Silver
in his Reinforcement Learning Lecture3 - Planning by Dynamic 
Programming. 
Author: Qiang Ye
Date: July 1, 2017

The value function converges to:
 0.00 -14.00 -20.00 -22.00 
-14.00 -18.00 -20.00 -20.00 
-20.00 -20.00 -18.00 -14.00 
-22.00 -20.00 -14.00   0.00 
At Iterate No.153
'''
# id of the states, 0 and 15 are terminal states
states = [i for i in range(16)]
#  0* 1  2   3  
#  4  5  6   7
#  8  9  10  11
#  12 13 14  15*

# initial values of states
values = [0  for _ in range(16)]

# Action
actions = ["n", "e", "s", "w"]

# 行为对应的状态改变量
# use a dictionary for convenient computation of next state id.
ds_actions = {"n": -4, "e": 1, "s": 4, "w": -1}  

# discount factor
gamma = 1.00

# random move policy
# def myAction():
#   return actions[randint(0,3)]

# compute next state id and corresponding reward according to 
# current state and action.
# 根据当前状态和采取的行为计算下一个状态id以及得到的即时奖励
def nextState(s, a):
  next_state = s
  if (s%4 == 0 and a == "w") or (s<4 and a == "n") or \
     ((s+1)%4 == 0 and a == "e") or (s > 11 and a == "s"):
    pass
  else:
    ds = ds_actions[a]
    next_state = s + ds
  return next_state

# reward of a state
def rewardOf(s):
  return 0 if s in [0,15] else -1
  
# check if a state is terminate state
def isTerminateState(s):
  return s in [0,15]

# get successor states of a given state s
def getSuccessors(s):
  successors = []
  if isTerminateState(s):
    return successors
  for a in actions:
    next_state = nextState(s, a)
    # if s != next_state:
    successors.append(next_state)
  return successors

# update the value of state s
def updateValue(s):
  sucessors = getSuccessors(s)
  newValue = 0  # values[s]
  num = 4       # len(successors)
  reward = rewardOf(s)
  for next_state in sucessors:
    newValue += 1.00/num * (reward + gamma * values[next_state])
  return newValue
    
# perform one-step iteration
def performOneIteration():
  newValues = [0 for _ in range(16)]
  for s in states:
    newValues[s] = updateValue(s)
  global values
  values = newValues
  printValue(values)

# show some array info of the small grid world
def printValue(v):
  for i in range(16):
    print('{0:>6.2f}'.format(v[i]),end = " ")
    if (i+1)%4 == 0:
      print("")
  print()
      
# test function
def test():
  printValue(states)
  printValue(values)
  for s in states:
    reward = rewardOf(s)
    for a in actions:
      next_state = nextState(s, a)
      print("({0}, {1}) -> {2}, with reward {3}".format(s, a,next_state, reward))

  for i in range(200):
    performOneIteration()
    printValue(values)
    
def main():
  max_iterate_times = 160
  cur_iterate_times = 0
  while cur_iterate_times <= max_iterate_times:
    print("Iterate No.{0}".format(cur_iterate_times))
    performOneIteration()
    cur_iterate_times += 1
  printValue(values)

if __name__ == '__main__':
  main()
  
  
