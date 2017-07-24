import argparse
import random
import numpy as np
import gym
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box


from collections import namedtuple
from keras import layers
from keras import models
from keras import backend as K
from keras import optimizers
from keras import regularizers
from typing import List


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--l2",
                    type=float,
                    dest="L2",
                    default=0.01,
                    help="L2 Regularizer")
parser.add_argument("--hidden",
                    type=int,
                    dest="HIDDEN",
                    default=200,
                    help="Number of hidden units")
parser.add_argument("--capacity",
                    type=int,
                    dest="CAPACITY",
                    default=1000000,
                    help="Capacity for ReplayMemory")
parser.add_argument("--n_episode",
                    type=int,
                    dest='N_EPISODE',
                    default=1000,
                    help="Number of episodes to train")
parser.add_argument("--batch-size",
                    type=int,
                    default=16, # 64
                    dest="BATCH_SIZE",
                    help="Size of minibatch")
parser.add_argument("--gamma",
                    type=float,
                    default=0.99,
                    dest="GAMMA",
                    help="Gamma, the discount rate")
parser.add_argument("--tau",
                    type=float,
                    dest="TAU",
                    default=0.001,
                    help="Tau for soft update (the lower the softer update)")
parser.add_argument("--env",
                    type=str,
                    dest="ENV",
                    default="Pendulum-v0",
                    help="Name of gym environment")
parser.add_argument("--use-batchnorm",
                    action="store_true",
                    dest="USE_BATCHNORM",
                    help="Use batchnorm")
parser.add_argument("--clip-value",
                    type=float,
                    default=40,
                    dest="CLIP_VALUE",
                    help="Gradient clipping value (positive)")
FLAGS = parser.parse_args()

nfh = 20    
                # 网络隐藏层神经元数量
Transition = namedtuple("Transition", field_names=["state",
                                              "action",
                                              "reward",
                                              "is_done",
                                              "next_state"])

class Episode(object):
    def __init__(self, name=None) -> None:
        self.name = name                # 名称
        self.t_reward = 0           # 总的获得的奖励
        self.len = 0                 # episode长度
        self.trans_list = []            # 一次状态转移
        self.cur_pos = -1               # 当前位置

    def push(self, s0: np.ndarray, 
                   a0: np.ndarray, 
                   r: np.ndarray, 
                   is_done: bool, 
                   s1: np.ndarray) -> None:
        s0, a0, s1 = np.ravel(s0), np.squeeze(a0), np.ravel(s1)
        self.trans_list.append(Transition(s0, a0, r, is_done, s1))
        self.len += 1

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

class Experience(object):
    def __init__(self, capacity: int, name=None) -> None:
        """Creates a Experience with given `capacity`
        Args:
            capacity (int): Max capacity of the experience
        """
        self.name = name
        self.capacity = capacity
        self.cur_pos = 0
        self.episodes = []

    def push(self, episode: Episode) -> None:
        """Stores values to the experience by creating a `Transition`
        Args:
            episode
        """
        if len(self.episodes) < self.capacity:
            self.episodes.append(None)

        self.episode[self.cur_pos] = episode
        self.cur_pos = (self.cur_pos + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        """Returns a mini-batch of the experience
        Args:
            batch_size (int): Size of mini-batch
        Returns:
            List[Transition]: Mini-batch of `Transitions`
        """
        return random.sample(self.episodes, k=batch_size)

    def __len__(self) -> int:
        """Returns the current size of the experience
        Returns:
            int: Current size of the experience
        """
        return len(self.episodes)

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

    def push(self, s: np.ndarray, a: np.ndarray, r: np.ndarray, d: bool, s2: np.ndarray) -> None:
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

class ActorNet(object):
    '''构建一个确定型策略网络：ActorNet,针对离散型行为
    '''
    def __init__(self, nfs: int, 
                       nfh: int, 
                       nfa: int,
                       a_value_bound: float,
                       use_batchnorm: bool) -> None:
        # 指定输入、输出特征数，是否使用batchnorm
        self.nfs, self.nfh, self.nfa = nfs, nfh, nfa
        self.a_value_bound = a_value_bound
        self.use_batchnorm = use_batchnorm
        self._build_network()

    def _build_network(self) -> None:
        states = layers.Input(shape=(self.nfs,), name="states")

        # Layer1: state -> (bn) -> relu
        #net = layers.Dense(units=self.nfh,
        #                   kernel_regularizer=regularizers.l2(FLAGS.L2))(states)
        #if self.use_batchnorm:
        #    net = layers.BatchNormalization()(net)
        #net = layers.Activation("relu")(net)

        # Layer2: Layer1 -> (bn) -> relu
        net = layers.Dense(units=self.nfh,
                           kernel_regularizer=regularizers.l2(FLAGS.L2))(states)
        if self.use_batchnorm:
            net = layers.BatchNormalization()(net)
        #net = layers.Activation("relu")(net)

        # Layer3: Layer2 -> tanh -> actions -> actions_scaled
        net = layers.Dense(units=self.nfa,
                           kernel_regularizer=regularizers.l2(FLAGS.L2))(net)
        actions = layers.Activation("tanh")(net)
        actions = layers.Lambda(lambda x: x * self.a_value_bound)(actions)

        self.model = models.Model(inputs=states, outputs=actions)
        
        action_grad = layers.Input(shape=(self.nfa,))
        loss = K.mean(-action_grad * actions)

        for l2_regularizer_loss in self.model.losses:
            loss += l2_regularizer_loss

        optimizer = optimizers.Adam(clipvalue=FLAGS.CLIP_VALUE)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights,
                                           constraints=self.model.constraints,
                                           loss=loss)

        self.train_fn = K.function(inputs=[self.model.input, 
                                           action_grad, 
                                           K.learning_phase()],
                                   outputs=[],
                                   updates=updates_op)

        
        return

class CriticNet(object):
    """Critic Network is basically a function such that
        f(state, action) -> value (= Q_value)
    """    
    def __init__(self, nfs: int, nfa: int, use_batchnorm: bool) -> None:
        """Builds a network at initialization
        Args:
            nfs (int): Dimension of `state`
            nfa (int): Dimension of `action`
            use_batchnorm (bool): Use BatchNormalization if `True`
        """
        self.nfs = nfs
        self.nfa = nfa
        self.use_batchnorm = use_batchnorm
        self._build_network()

    def _build_network(self) -> None:
        """Critic Network Architecture
        (1) [states] -> fc -> (bn) -> relu -> fc
        (2) [actions] -> fc
        (3) Merge[(1) + (2)] -> (bn) -> relu
        (4) [(3)] -> fc (= Q_pred)
        Notes:
            `Q_grad` is `d_Q_pred/d_action` required for `ActorNetwork`
        """
        states = layers.Input(shape=(self.nfs,), name="states")
        actions = layers.Input(shape=(self.nfa,), name="actions")

        # Layer 1: states -> fc -> (bn) -> relu
        #net = layers.Dense(units=FLAGS.HIDDEN,
        #                   kernel_regularizer=regularizers.l2(FLAGS.L2))(states)
        #if self.use_batchnorm:
        #    net = layers.BatchNormalization()(net)
        #net = layers.Activation("relu")(net)

        # Layer 2:
        # Merge[Layer1 -> fc, actions -> fc] -> (bn) -> relu
        states_out = layers.Dense(units=FLAGS.HIDDEN,
                                  kernel_regularizer=regularizers.l2(FLAGS.L2)
                                  )(states)
        actions_out = layers.Dense(units=FLAGS.HIDDEN,
                                   kernel_regularizer=regularizers.l2(FLAGS.L2)
                                   )(actions)
        net = layers.Add()([states_out, actions_out])
        if self.use_batchnorm:
            net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)

        # Layer 3: Layer2 -> fc
        Q_pred = layers.Dense(units=1,
                              kernel_regularizer=regularizers.l2(FLAGS.L2))(net)

        Q_grad = K.gradients(Q_pred, actions)
        self.model = models.Model(inputs=[states, actions], outputs=Q_pred)

        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss="mse")
        self.get_Q_grad = K.function(inputs=[*self.model.input, 
                                             K.learning_phase()],
                                     outputs=Q_grad)



class ActorCriticAgent(object):
    def __init__(self, env: gym.Env) -> None:
        """Creates 4 networks in total
        (1) Critic Local & Target network
        (2) Actor Local & Target network
        **Target networks are soft-updated**
        Args:
            nfs (int): Dimension of `state`
            nfa (int): Dimenionf of `action`
            a_value_bound (float, optional): Max value for `action` (upper bound)
            use_batchnorm (bool, optional): Batch Normalization
        """
        self.a_space_type = None
        self.nfh = nfh
        self.nfs = env.observation_space.shape[0]
        if isinstance(env.action_space, Discrete):
            self.a_space_type = "discrete"
            self.a_value_bound = 1.0
            self.nfa = env.action_space.n
        elif isinstance(env.action_space, Box):
            self.a_space_type = "continuous"
            self.nfa = env.action_space.shape[0]
            self.a_value_bound = env.action_space.high
        self.use_batchnorm = False # True

        self.c_local_net = CriticNet(self.nfs, self.nfa, self.use_batchnorm)
        self.c_tgt_net = CriticNet(self.nfs, self.nfa, self.use_batchnorm)

        self.a_local_net = ActorNet(self.nfs, 
                                    self.nfh,
                                    self.nfa, 
                                    self.a_value_bound, 
                                    self.use_batchnorm)
        self.a_tgt_net = ActorNet(self.nfs, 
                                  self.nfh,
                                  self.nfa, 
                                  self.a_value_bound, 
                                  self.use_batchnorm)
        self.noise = OUNoise(self.nfa)
        self._initialize_target()

    def perform_policy(self, states: np.ndarray) -> np.ndarray:
        """Returns actions given `states`
        Args:
            states (np.ndarray): `state` array, shape (n, nfs)
        Returns:
            np.ndarray: `action` array, shape (n, action_dim)
        """
        states = np.reshape(states, [-1, self.nfs])
        actions = self.a_local_net.model.predict(states)
        if self.a_space_type == "discrete":
            probs = np.squeeze(K.eval(K.softmax(actions)))
            # print(probs[0])
            # print(type(probs))
            actions = np.zeros(self.nfa,dtype="float32")
            prob = random.random()
            total_p = 0.00
            for i,p in enumerate(probs):
                total_p += p
                if (total_p >= prob):
                    actions[i] = 1.0
                    return actions
            actions[len(probs)-1] = 1.0
            return actions
            # print("probs:{0}".format(probs))
        else:
            return actions + self.noise.noise()

    def learning(self, episodes: List[Episode], gamma: float, tau: float) -> None:
        """Update parameters
        y = r + γ * Q_target(s_next, Actor_target(s_next))
        where
            Actor_target(state) -> action
            Q_target(state, action) -> value
        Args:
            episodes (List[Transition]): Minibatch from `ReplayMemory`
            gamma (float): Discount rate for Q_target
            tau (float): Soft learning parameter
        """
        states = np.vstack([e.state for e in episodes if e is not None])
        actions = np.array([e.action for e in episodes if e is not None])\
                        .astype(np.float32).reshape(-1, self.nfa)
        rewards = np.array([e.reward for e in episodes if e is not None])\
                    .astype(np.float32).reshape(-1, 1)
        dones = np.array([e.is_done for e in episodes if e is not None])\
                    .astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in episodes if e is not None])

        target_actions = self.a_tgt_net.model.predict_on_batch(next_states)
        target_values = self.c_tgt_net.model.predict_on_batch([next_states, 
                                                               target_actions])

        Q_target = rewards + gamma * target_values * (1 - dones)

        assert target_values.shape[1] == 1, target_values.shape
        assert Q_target.shape[1] == 1, Q_target.shape

        self.c_local_net.model.train_on_batch(x=[states, actions], y=Q_target)

        Q_grad = np.reshape(self.c_local_net.get_Q_grad([states, actions, 0]), 
                            (-1, self.nfa))

        assert Q_grad.shape[1] == self.nfa, Q_grad.shape

        self.a_local_net.train_fn([states, Q_grad, 1])

        self._soft_update(self.c_local_net.model, self.c_tgt_net.model, tau)
        self._soft_update(self.a_local_net.model, self.a_tgt_net.model, tau)

    def _initialize_target(self) -> None:
        """Set target parameters equal to the local parameters
        """
        self.c_tgt_net.model.set_weights(self.c_local_net.model.get_weights())
        self.a_tgt_net.model.set_weights(self.a_local_net.model.get_weights())

    def _soft_update(self, 
                    local_model: models.Model, 
                    target_model: models.Model, 
                    tau: float) -> None:
        """Soft update parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Args:
            local_model (models.Model): Keras model (weights will be copied from)
            target_model (models.Model): Keras model (weights will be copied to)
            tau (float): Decides how much local values should be updated
        """
        target_weights = np.array(target_model.get_weights())
        local_weights = np.array(local_model.get_weights())

        assert len(target_weights) == len(local_weights)

        new_weights = tau * local_weights + (1 - tau) * target_weights
        target_model.set_weights(new_weights)

class OUNoise:
    """Ornstein-Uhlenbeck process
    Attributes:
        action_dimension (int): Dimension of `action`
        mu (float): 0.0
        sigma (float): > 0
        state (np.ndarray): Noise
        theta (float): > 0
    Notes:
        https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """

    def __init__(self, action_dimension: int, 
                       mu=0.0, 
                       theta=0.15, 
                       sigma=0.3, 
                       seed=123) -> None:
        """Initializes the noise """
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()
        np.random.seed(seed)

    def reset(self) -> None:
        """Resets the states(= noise) to mu
        """
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self) -> np.ndarray:
        """Returns a noise(= states)
        Returns:
            np.ndarray: noise, shape (n, action_dim)
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


def train_episode(env: gym.Env, 
                  agent: ActorCriticAgent, 
                  memory: ReplayMemory,
                  batch_size: int,
                  gamma: float,
                  tau: float
                 ) -> float:
    s = env.reset()
    done = False

    total_reward = 0

    while not done:
        a = agent.perform_policy(s)
        if agent.a_space_type == "discrete":
            discrete_a = np.argmax(a)
            s2, r, done, info = env.step(discrete_a)
        else:
            s2, r, done, info = env.step(a)    
        env.render()        
        memory.push(s, a, r, done, s2)

        total_reward += r

        if len(memory) > batch_size:

            transition_batch = memory.sample(batch_size)
            agent.learning(transition_batch, gamma, tau)

        s = s2

        if done:
            return total_reward 
        #print(total_reward)

def main() -> None:
    import puckworld as pw
    env = gym.make("PuckWorld-v0")
    env = gym.wrappers.Monitor(env, directory='monitors', force=True)
    try:
        

        # env = gym.wrappers.Monitor(env, directory='monitors', force=True)

        agent = ActorCriticAgent(env)
        
        memory = ReplayMemory(FLAGS.CAPACITY)

        for episode_i in range(FLAGS.N_EPISODE):
            reward = train_episode(env, 
                                   agent, 
                                   memory, 
                                   FLAGS.BATCH_SIZE, 
                                   FLAGS.GAMMA, 
                                   FLAGS.TAU)
            print(episode_i, reward)
    finally:
        env.close()


if __name__ == '__main__':
    main()