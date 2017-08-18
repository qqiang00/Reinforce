# Learn reinforcement learning with classic GridWorld and PuckWorld Environments compatible with Gym library.
I wrote several basic classes describing the events occured during an agent's interaction with an environment. Besides, for RL beginners to better understand how the classic RL algorithms work in discrete observation spaces, I wrote two classic environments:GridWorld and PuckWorld. 

You can copy these two environments into your gym library and by just making a few modification, these two environments can be used the same as the embeded environments in Gym.

Here is the organization of the whole package:

## [core.py](https://github.com/qqiang00/reinforce/blob/master/reinforce/core.py)
You will find some core classes modeling the object needed in reinforcement learning in this file. These are:

### Transition
stores the information describing an agent's state transition. Transition is the basic unit of an Episode.

### Episode
stores a list of transition that an agent experience till to one of its end states.

### Experience
stores a list of episode. Experience has a capacity limit; it also has a sample method to randomly select a certain number of transitions from its memory.

### Agent
this is the base class for all agents implemented for a certain reinforcement learning algorithm. in Agent class, an "act" function wraps the step() function of an environment which interacts with the agent. you can implement your own agent class by deriving this class.

## [agents.py](https://github.com/qqiang00/reinforce/blob/master/reinforce/agents.py)
In this file, you will find some agents class which are already implemented for a certain reinforcement learning algorithms. more agents classes will be added into this file as I practice. Now, you can find agent with sarsa, Q, sarsa(\lambda) algorithms.

## [approximator.py](https://github.com/qqiang00/reinforce/blob/master/reinforce/approximator.py)
You can find some classes which performs like a neural network. that's right. Deep neural network is used as an function approximator in RL algorithms, this is so called Deep reinforcement Learning. You will find different types of Agents using different type of function approximators.

## [gridworld.py](https://github.com/qqiang00/reinforce/blob/master/reinforce/gridworld.py)
A base GridWorld classe is implemented for generating more specific GridWorld environments used in David Silver's RL course, such as:
* Simple 10×7 Grid world
* Windy Grid world
* Random Walk
* Cliff Walk
* Skull and Treasure Environment used for explain an agent can benefit from random policy, while a determistic policy may lead to an endless loop.

You can build your own grid world object just by giving different parameters to its init function. 
Visit [here](https://zhuanlan.zhihu.com/p/28109312) for more details about how to generate a specific grid world environment object.

## [puckworld.py](https://github.com/qqiang00/reinforce/blob/master/reinforce/puckworld.py)
This is another classic environment called "PuckWorld", the idea of which comes from [ReinforceJS](http://cs.stanford.edu/people/karpathy/reinforcejs/puckworld.html). Thanks to Karpathy.
Different from gridworld environment which has a one-dimensional discrete observation and action space, puck world has a continuous observation state space with six dimensions and a discrete action space which can also easily be converted to continuous one. 

PuckWord is considered as one of the classic environments for training an agent with Deep Q-Learning Network.

## [examples](https://github.com/qqiang00/reinforce/tree/master/reinforce/examples)
several seperate .pys are provided for understanding a RL algorithm without the classes mentioned above. 

you can also find a implementation of Policy Iteration and Value Iteration by using dynamic programming in this folder.


Hope you enjoy these classes and expect you to make contribution for this package.

##
Author: Qiang Ye.

Date: August 16, 2017

License: MIT

