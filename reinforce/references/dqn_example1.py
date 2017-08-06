"""
DQN (NIPS 2013)
Playing Atari with Deep Reinforcement Learning
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
"""
import numpy as np
import tensorflow as tf
import random
import dqn
import gym
from collections import deque

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, 'gym-results/', force=True)
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
MAX_EPISODE = 5000
BATCH_SIZE = 64

# minimum epsilon for epsilon greedy
MIN_E = 0.0
# epsilon will be `MIN_E` at `EPSILON_DECAYING_EPISODE`
EPSILON_DECAYING_EPISODE = MAX_EPISODE * 0.01


def bot_play(mainDQN: dqn.DQN) -> None:
    """Runs a single episode with rendering and prints a reward
    Args:
        mainDQN (dqn.DQN): DQN Agent
    """
    state = env.reset()
    total_reward = 0

    while True:
        env.render()
        action = np.argmax(mainDQN.predict(state))
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            print("Total score: {}".format(total_reward))
            break


def train_minibatch(DQN: dqn.DQN, train_batch: list) -> float:
    """Prepare X_batch, y_batch and train them
    Recall our loss function is
        target = reward + discount * max Q(s',a)
                 or reward if done early
        Loss function: [target - Q(s, a)]^2
    Hence,
        X_batch is a state list
        y_batch is reward + discount * max Q
                   or reward if terminated early
    Args:
        DQN (dqn.DQN): DQN Agent to train & run
        train_batch (list): Minibatch of Replay memory
            Eeach element is a tuple of (s, a, r, s', done)
    Returns:
        loss: Returns a loss
    """
    state_array = np.vstack([x[0] for x in train_batch])
    action_array = np.array([x[1] for x in train_batch])
    reward_array = np.array([x[2] for x in train_batch])
    next_state_array = np.vstack([x[3] for x in train_batch])
    done_array = np.array([x[4] for x in train_batch])

    X_batch = state_array
    y_batch = DQN.predict(state_array)

    Q_target = reward_array + DISCOUNT_RATE * np.max(DQN.predict(next_state_array), axis=1) * ~done_array
    y_batch[np.arange(len(X_batch)), action_array] = Q_target

    # Train our network using target and predicted Q values on each episode
    loss, _ = DQN.update(X_batch, y_batch)

    return loss


def annealing_epsilon(episode: int, min_e: float, max_e: float, target_episode: int) -> float:
    """Return an linearly annealed epsilon
    Epsilon will decrease over time until it reaches `target_episode`
         (epsilon)
             |
    max_e ---|\
             | \
             |  \
             |   \
    min_e ---|____\_______________(episode)
                  |
                 target_episode
     slope = (min_e - max_e) / (target_episode)
     intercept = max_e
     e = slope * episode + intercept
    Args:
        episode (int): Current episode
        min_e (float): Minimum epsilon
        max_e (float): Maximum epsilon
        target_episode (int): epsilon becomes the `min_e` at `target_episode`
    Returns:
        float: epsilon between `min_e` and `max_e`
    """

    slope = (min_e - max_e) / (target_episode)
    intercept = max_e

    return max(min_e, slope * episode + intercept)


def main():
    # store the previous observations in replay memory
    replay_buffer = deque(maxlen=REPLAY_MEMORY)
    last_100_game_reward = deque(maxlen=100)

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE)
        init = tf.global_variables_initializer()
        sess.run(init)

        for episode in range(MAX_EPISODE):
            e = annealing_epsilon(episode, MIN_E, 1.0, EPSILON_DECAYING_EPISODE)
            done = False
            state = env.reset()

            step_count = 0
            while not done:

                if np.random.rand() < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                next_state, reward, done, _ = env.step(action)

                if done:
                    reward = -1

                replay_buffer.append((state, action, reward, next_state, done))

                state = next_state
                step_count += 1

                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    train_minibatch(mainDQN, minibatch)

            print("[Episode {:>5}]  steps: {:>5} e: {:>5.2f}".format(episode, step_count, e))

            # CartPole-v0 Game Clear Logic
            last_100_game_reward.append(step_count)
            if len(last_100_game_reward) == last_100_game_reward.maxlen:
                avg_reward = np.mean(last_100_game_reward)
                if avg_reward > 199.0:
                    print("Game Cleared within {} episodes with avg reward {}".format(episode, avg_reward))
                    break


if __name__ == "__main__":
    main()