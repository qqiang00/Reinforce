"""
Deep Deterministic Policy Gradients (DDPG)
https://arxiv.org/pdf/1509.02971.pdf
TODO: Batch Normalization Bug
"""
import argparse
import random
import numpy as np
import gym

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
                    default=64,
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


class ActorNetwork(object):
    """Actor Network is a deterministic policy such that
        f(states) -> action
    """

    def __init__(self, input_dim: int, output_dim: int, action_bound: float, use_batchnorm: bool) -> None:
        """Builds a network
        Args:
            input_dim (int): Dimension of `state`
            output_dim (int): Dimension of `action`
            action_bound (float): Max Value of `action`
            use_batchnorm (bool): Use Batchnormalization
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.action_bound = action_bound
        self.use_batchnorm = use_batchnorm

        self._build_network()

    def _build_network(self) -> None:
        """Actor network architecture
        Notes:
            Network: 𝜇(state) -> continuous action
            Loss: Policy Gradient
                mean(d_Q(s, a)/d_a * d_𝜇(s)/d_𝜃) + L2_Reg
        """
        states = layers.Input(shape=(self.input_dim,), name="states")

        # Layer1: state -> (bn) -> relu
        net = layers.Dense(units=FLAGS.HIDDEN,
                           kernel_regularizer=regularizers.l2(FLAGS.L2))(states)
        if self.use_batchnorm:
            net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)

        # Layer2: Layer1 -> (bn) -> relu
        net = layers.Dense(units=FLAGS.HIDDEN,
                           kernel_regularizer=regularizers.l2(FLAGS.L2))(net)
        if self.use_batchnorm:
            net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)

        # Layer3: Layer2 -> tanh -> actions -> actions_scaled
        net = layers.Dense(units=self.output_dim,
                           kernel_regularizer=regularizers.l2(FLAGS.L2))(net)
        actions = layers.Activation("tanh")(net)
        actions = layers.Lambda(lambda x: x * self.action_bound)(actions)

        self.model = models.Model(inputs=states, outputs=actions)

        action_grad = layers.Input(shape=(self.output_dim,))
        loss = K.mean(-action_grad * actions)

        for l2_regularizer_loss in self.model.losses:
            loss += l2_regularizer_loss

        optimizer = optimizers.Adam(clipvalue=FLAGS.CLIP_VALUE)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights,
                                           constraints=self.model.constraints,
                                           loss=loss)

        self.train_fn = K.function(inputs=[self.model.input, action_grad, K.learning_phase()],
                                   outputs=[],
                                   updates=updates_op)


class CriticNetwork(object):
    """Critic Network is basically a function such that
        f(state, action) -> value (= Q_value)
    """

    def __init__(self, input_dim: int, output_dim: int, use_batchnorm: bool) -> None:
        """Builds a network at initialization
        Args:
            input_dim (int): Dimension of `state`
            output_dim (int): Dimension of `action`
            use_batchnorm (bool): Use BatchNormalization if `True`
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
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
        states = layers.Input(shape=(self.input_dim,), name="states")
        actions = layers.Input(shape=(self.output_dim,), name="actions")

        # Layer 1: states -> fc -> (bn) -> relu
        net = layers.Dense(units=FLAGS.HIDDEN,
                           kernel_regularizer=regularizers.l2(FLAGS.L2))(states)
        if self.use_batchnorm:
            net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)

        # Layer 2:
        # Merge[Layer1 -> fc, actions -> fc] -> (bn) -> relu
        states_out = layers.Dense(units=FLAGS.HIDDEN,
                                  kernel_regularizer=regularizers.l2(FLAGS.L2))(net)
        actions_out = layers.Dense(units=FLAGS.HIDDEN,
                                   kernel_regularizer=regularizers.l2(FLAGS.L2))(actions)
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
        self.get_Q_grad = K.function(inputs=[*self.model.input, K.learning_phase()],
                                     outputs=Q_grad)


def soft_update(local_model: models.Model, target_model: models.Model, tau: float) -> None:
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


Transition = namedtuple("Transition",
                        field_names=["state", "action", "reward", "done", "next_state"])


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


class Agent(object):
    """Agent class
    (1) initialize local and target networks
    (2) get actions
    (3) update parameters
    Attributes:
        action_bound (float): Max value for `action`
        actor_local (ActorNetwork): Local network (updated by `Adam`)
        actor_target (ActorNetwork): Target network (soft updated after every training step)
        critic_local (CriticNetwork): Local network (updated by `Adam`)
        critic_target (CriticNetwork): Target network (soft updated after every training step)
        input_dim (int): Dimension of `state`
        noise (OUNoise): Will add an exploration noise
        output_dim (int): Dimension of `action`
        use_batchnorm (bool): If True, use BatchNormalization
    """

    def __init__(self, input_dim: int, output_dim: int, action_bound=1.0, use_batchnorm=True) -> None:
        """Creates 4 networks in total
        (1) Critic Local & Target network
        (2) Actor Local & Target network
        **Target networks are soft-updated**
        Args:
            input_dim (int): Dimension of `state`
            output_dim (int): Dimenionf of `action`
            action_bound (float, optional): Max value for `action` (upper bound)
            use_batchnorm (bool, optional): Batch Normalization
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.action_bound = action_bound
        self.use_batchnorm = use_batchnorm

        self.critic_local = CriticNetwork(self.input_dim, self.output_dim, self.use_batchnorm)
        self.critic_target = CriticNetwork(self.input_dim, self.output_dim, self.use_batchnorm)

        self.actor_local = ActorNetwork(self.input_dim, self.output_dim, self.action_bound, self.use_batchnorm)
        self.actor_target = ActorNetwork(self.input_dim, self.output_dim, self.action_bound, self.use_batchnorm)

        self.noise = OUNoise(self.output_dim)

    def get_actions(self, states: np.ndarray) -> np.ndarray:
        """Returns actions given `states`
        Args:
            states (np.ndarray): `state` array, shape (n, input_dim)
        Returns:
            np.ndarray: `action` array, shape (n, action_dim)
        """
        states = np.reshape(states, [-1, self.input_dim])
        actions = self.actor_local.model.predict(states)

        return actions + self.noise.noise()

    def update(self, transitions: List[Transition], gamma: float, tau: float) -> None:
        """Update parameters
        y = r + γ * Q_target(s_next, Actor_target(s_next))
        where
            Actor_target(state) -> action
            Q_target(state, action) -> value
        Args:
            transitions (List[Transition]): Minibatch from `ReplayMemory`
            gamma (float): Discount rate for Q_target
            tau (float): Soft update parameter
        """
        states = np.vstack([t.state for t in transitions if t is not None])
        actions = np.array([t.action for t in transitions if t is not None]).astype(np.float32).reshape(-1, self.output_dim)
        rewards = np.array([t.reward for t in transitions if t is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([t.done for t in transitions if t is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([t.next_state for t in transitions if t is not None])

        target_actions = self.actor_target.model.predict_on_batch(next_states)
        target_values = self.critic_target.model.predict_on_batch([next_states, target_actions])

        Q_target = rewards + gamma * target_values * (1 - dones)

        assert target_values.shape[1] == 1, target_values.shape
        assert Q_target.shape[1] == 1, Q_target.shape

        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_target)

        Q_grad = np.reshape(self.critic_local.get_Q_grad([states, actions, 0]), (-1, self.output_dim))

        assert Q_grad.shape[1] == self.output_dim, Q_grad.shape

        self.actor_local.train_fn([states, Q_grad, 1])

        soft_update(self.critic_local.model, self.critic_target.model, tau)
        soft_update(self.actor_local.model, self.actor_target.model, tau)

    def initialize_target(self) -> None:
        """Set target parameters equal to the local parameters
        """
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())


def train_episode(env: gym.Env, agent: Agent, memory: ReplayMemory, batch_size: int, gamma: float, tau: float) -> float:
    """Runs an episode and train
    Args:
        env (gym.Env): gym Environment
        agent (Agent): `agent` will perform all the dirty step
        memory (ReplayMemory): `ReplayMemory`
        batch_size (int): Minibatch size
        gamma (float): Discount Rate for target Q
        tau (float): Soft update parameters
    Returns:
        float: Total reward from this episode
    """
    s = env.reset()
    done = False

    total_reward = 0

    while not done:
        a = agent.get_actions(s)
        s2, r, done, info = env.step(a)
        memory.push(s, a, r, done, s2)

        total_reward += r

        if len(memory) > batch_size:

            transition_batch = memory.sample(batch_size)
            agent.update(transition_batch, gamma, tau)

        s = s2

        if done:
            return total_reward


def get_env_dim(env: gym.Env) -> tuple:
    """Returns `input_dim` and `output_dim`
    Args:
        env (gym.Env): Gym Environment should be declared prior
    Returns:
        int: Dimension of `state`
        int: Dimension of `action`
    """
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]

    return input_dim, output_dim


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

    def __init__(self, action_dimension: int, mu=0.0, theta=0.15, sigma=0.3, seed=123) -> None:
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


def main() -> None:
    """MAIN """

    try:
        env = gym.make(FLAGS.ENV)
        env = gym.wrappers.Monitor(env, directory='monitors', force=True)

        input_dim, output_dim = get_env_dim(env)
        action_bound = env.action_space.high
        np.testing.assert_almost_equal(action_bound * -1, env.action_space.low)

        print(env.observation_space)
        print(env.action_space)

        agent = Agent(input_dim, output_dim, action_bound, FLAGS.USE_BATCHNORM)
        agent.initialize_target()

        memory = ReplayMemory(FLAGS.CAPACITY)

        for episode_i in range(FLAGS.N_EPISODE):
            reward = train_episode(env, agent, memory, FLAGS.BATCH_SIZE, FLAGS.GAMMA, FLAGS.TAU)
            print(episode_i, reward)

    finally:
        env.close()


if __name__ == '__main__':
    main()