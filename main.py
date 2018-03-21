#!/usr/bin/env python
### TODO

import matplotlib as mpl
import keras
import numpy as np, gym, sys, copy, argparse
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import optimizers
from keras.engine.topology import Layer

import random
import time

DUEL = 2
DEEP = 1


class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, env, agent, deep=0):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.

        """
        self.board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
            batch_size=32, write_graph=True, write_grads=False, write_images=False,
            embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        """

        self.model = None
        self.agent = agent

        self.state_size = env.observation_space.low.size
        self.action_size = env.action_space.n

        self.deep = deep

        def regularizer(x):
            x -= keras.backend.mean(x, axis=1, keepdims=True)  # action column
            return x

        def regdim(xs):
            return xs

        if deep == DUEL:
            input_layer = keras.layers.Input(shape=(self.state_size + 1,), name='input')
            dense = keras.layers.Dense(10, activation='relu')(input_layer)
            dense = keras.layers.Dense(10, activation='relu')(dense)
            dense = keras.layers.Dense(10, activation='relu')(dense)
            denseA = keras.layers.Dense(10, activation='relu')(dense)
            denseV = keras.layers.Dense(10, activation='relu')(dense)

            outA = keras.layers.Dense(self.action_size, activation='linear')(denseA)
            x = keras.layers.Lambda(regularizer, output_shape=regdim)(outA)

            outV = keras.layers.Dense(1, activation='linear')(denseV)
            y = keras.layers.RepeatVector(self.action_size)(outV)
            y_prime = keras.layers.Reshape((self.action_size,))(y)

            finalOut = keras.layers.Add()([x, y_prime])
            model = keras.models.Model(inputs=[input_layer], outputs=[finalOut])
            adam = optimizers.Adam()  # lr=self.agent.alpha, decay=1e-6)
            model.compile(loss='mse',
                          optimizer=adam,
                          metrics=['accuracy'])
            self.model = model

        elif deep == DEEP:
            model = Sequential()
            model.add(Dense(10, activation='relu', input_dim=(self.state_size + 1)))
            model.add(Dense(10, activation='relu'))
            model.add(Dense(10, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))

            adam = optimizers.Adam(lr=self.agent.alpha, decay=1e-6)
            model.compile(loss='mse',
                          optimizer=adam,
                          metrics=['accuracy'])
            self.model = model
        else:
            model = Sequential()
            model.add(Dense(self.action_size, activation='linear',
                            input_dim=self.state_size, use_bias=False))

            # adam = optimizers.Adam(lr=self.agent.alpha, decay=1e-6)
            # adam = optimizers.Nadam(lr=self.agent.alpha)
            adam = optimizers.Adam()
            # adam = optimizers.Adam(lr=1.0, decay=1e-3)
            model.compile(loss='mse',
                          optimizer=adam,
                          metrics=['accuracy'])
            self.model = model

    # states : np.array (num_inputs, num_dims)
    # return : np.array (num_inputs, num_actions)
    def predict(self, states):
        # We realized that keras automatically includes a bias term unless use_bias=False
        # is specified. However, to keep compatibility with our trained models, we are leaving this
        # in for the deep network models
        if self.deep != 0:
            bias_term = np.ones((states.shape[0], 1))
            states = np.append(states, bias_term, axis=1)

        return self.model.predict(states)

    # states : np.array (num_inputs, num_dims)
    # return best_action : (num_inputs, )
    # return best_action_value : (num_inputs, )
    def best_action_batch(self, states, terminals=None):
        next_state_actions = self.predict(states)
        best_actions = np.argmax(next_state_actions, axis=1)
        best_action_value = np.max(next_state_actions, axis=1)
        if terminals is not None:
            best_action_value[terminals] = 0.0
        return (best_actions, best_action_value)

    def best_action(self, state):
        actions, action_value = self.best_action_batch(np.array([state]))
        return (actions[0], action_value[0])

    def update_batch(self, size, experienceList):
        target_list = []
        cur_list = []

        next_states = np.array([experience[3] for experience in experienceList])
        rewards = np.array([experience[2] for experience in experienceList])
        states = np.array([experience[0] for experience in experienceList])
        actions = np.array([experience[1] for experience in experienceList])
        terminals = np.array([experience[4] for experience in experienceList])

        _, next_state_values = self.best_action_batch(next_states, terminals)
        new_targets = rewards + (self.agent.gamma * next_state_values)
        cur_targets = self.predict(states)

        # Basically sets cur_targets[actions[i]] = new_targets[i] for each i
        cur_targets[np.arange(cur_targets.shape[0]), actions] = new_targets

        if self.deep != 0:
            states = np.append(states, np.ones((states.shape[0], 1)), axis=1)
        self.model.fit(states, cur_targets, batch_size=size, verbose=0)

    def update(self, state, action, reward, next_state, is_terminal):
        self.update_batch(1, [[state, action, reward, next_state, is_terminal]])

    def save_model_weights(self, weight_file):
        # Helper function to save your model / weights.
        self.model.save_weights(weight_file)
        pass

    def load_model(self, model_file):
        # Helper function to load an existing model.
        self.model = keras.models.load_model(model_file)

    def load_model_weights(self, weight_file):
        # Helper function to load model weights.
        self.model.load_weights(weight_file)


class Replay_Memory():

    def __init__(self, agent, memory_size=50000, burn_in=10000):
        self.cache = [None for i in range(memory_size)]
        self.size = burn_in
        self.new = burn_in
        self.cap = memory_size
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.
        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

        # initialize burn in
        agent.burn_in_memory(self.cache, burn_in)

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        return random.sample(self.cache[:self.size], batch_size)

    def append(self, transition):
        self.cache[self.new] = transition
        self.size = min(self.size + 1, self.cap)
        self.new = (self.new + 1) % self.cap




class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #   (a) Epsilon Greedy Policy.
    #     (b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, render=False, use_replay=False,
                 deep=0, monitor=False):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.

        self.alpha = 0.0001
        self.epsilon = 0.5
        self.epsilon_target = 0.05
        self.epsilon_delta = (0.5 - 0.05) / 100000
        self.episodes = 1000000

        self.environment_name = environment_name
        self.deep = deep

        self.policynet = QNetwork(self.env, self, deep=deep)
        self.valuenet = QNetwork(self.env, self, deep = deep)

        self.targetnet = valuenet

        # self.qnet.load_model_weights(weight_file='BestLinearNoReplay')
        self.state_size = self.env.observation_space.low.size
        self.action_size = self.env.action_space.n

        if use_replay:
            # ia = init_Agent(environment_name, self)
            self.replay = Replay_Memory(self, burn_in=10000)

        self.render = render
        self.use_replay = use_replay

    # q_values: State * Action -> Value
    def epsilon_greedy_policy(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return self.greedy_policy(state)

    # epsilon greedy policy with custom epsilon
    def epsilon_greedy_policy2(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        return self.greedy_policy(state)

    # greedy policy
    def greedy_policy(self, state):
        best_action, _ = self.qnet.best_action(state)
        return best_action

    def train(self, episodes=5000):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.

        # Used for non experience replay mode
        iteration = 0
        update_queue = []

        testing_rewards = []
        max_weight = -200

        run_test = False

        for episode in range(episodes):


        for episode in range(episodes):
            cur_state = self.env.reset()

            while True:
                iteration += 1

                if iteration % 10000 == 0:
                    run_test = True

                self.epsilon -= self.epsilon_delta
                self.epsilon = max(self.epsilon, self.epsilon_target)
                if self.render:
                    self.env.render()
                    time.sleep(0.05)

                action = self.epsilon_greedy_policy(cur_state)
                next_state, reward, is_terminal, debug_info = self.env.step(action)
                is_terminal_win = (False if self.environment_name == 'MountainCar' and
                                            self.deep == 0 and next_state[0] < 0.5 else is_terminal)

                if self.use_replay:
                    REPLAY_BATCH = 16
                    if iteration % REPLAY_BATCH == 0:
                        batch = self.replay.sample_batch(32 * REPLAY_BATCH)
                        self.qnet.update_batch(32, batch)
                    self.replay.append([cur_state, action, reward, next_state, is_terminal_win])
                else:
                    if iteration % 32 == 0:
                        self.qnet.update_batch(32, update_queue)
                        update_queue = []
                    else:
                        update_queue.append([cur_state, action, reward, next_state, is_terminal_win])

                cur_state = next_state

                if is_terminal:
                    break

            if run_test:
                print(str(episode / 100) + ";" + str(iteration))
                _, average_reward, _ = self.test(20, self.epsilon_target)
                testing_rewards.append(average_reward)
                run_test = False
                if (average_reward > max_weight + 1):
                    self.qnet.save_model_weights("model" + str(episode / 10000) + "wt" + str(average_reward))
                    max_weight = average_reward
            if (episode % 1000 == 0):
                print(testing_rewards)
                _, average_reward, _ = self.test(20, self.epsilon_target)
                self.qnet.save_model_weights("modelfinal" + str(self.deep) + ";" +
                                             str(episode / 10000) + ";" + str(average_reward))

        _, avg, _ = self.test(100)
        return avg

    def test(self, episodes, epsilon=0.0, model_file=None):
        total = []
        reward_sum = 0.0
        for i in range(episodes):
            cur_state = self.env.reset()

            episode_reward = 0
            discount = 1
            while True:
                if self.render:
                    self.env.render()
                    time.sleep(0.05)

                action = self.epsilon_greedy_policy2(cur_state, epsilon)
                next_state, reward, is_terminal, debug_info = self.env.step(action)
                # self.qnet.update(cur_state, action, reward, next_state, is_terminal)
                cur_state = next_state

                episode_reward += discount * reward
                discount *= self.gamma
                if is_terminal:
                    break

            total.append(episode_reward)
            reward_sum += episode_reward

        average = reward_sum / episodes
        stddev = np.std(np.array(total))
        print("Test: " + str(average))
        return total, average, stddev

    def burn_in_memory(self, cache, bns):
        cur_state = self.env.reset()
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        for i in range(0, bns):
            action = self.epsilon_greedy_policy(cur_state)
            next_state, reward, is_terminal, debug_info = self.env.step(action)

            cache[i] = [cur_state, action, reward, next_state, is_terminal]
            # self.qnet.update(cur_state, action, reward, next_state, is_terminal)
            cur_state = next_state
            if is_terminal:
                cur_state = self.env.reset()

        # need to episode to finish or else monitor complains
        while True:
            action = self.epsilon_greedy_policy(cur_state)
            next_state, reward, is_terminal, debug_info = self.env.step(action)
            cur_state = next_state
            if is_terminal:
                break


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str, default='MountainCar')
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--deep', dest='deep', type=int, default=2)
    parser.add_argument('--replay', dest='replay', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str)
    parser.add_argument('--monitor', dest='monitor', type=int, default=0)
    return parser.parse_args()


def main(args):
    args = parse_arguments()
    environment_name = args.env

    try:
        # Setting the session to allow growth, so it doesn't allocate all GPU memory.
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        gpu_ops = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_ops)
        sess = tf.Session(config=config)

        # Setting this as the default tensorflow session.
        keras.backend.tensorflow_backend.set_session(sess)
    except NameError:
        pass

    '''
    if environment_name == 'MountainCar':
      avg = -200
      while avg == -200:
        agent = DQN_Agent(environment_name, render=args.render,
           use_replay=args.replay, deep=args.deep, monitor=args.monitor)
        avg = agent.train(10000)
      print("Found")
    elif environment_name == 'CartPole':
      agent = DQN_Agent(environment_name, render = args.render,
            use_replay=args.replay, deep=args.deep, monitor=args.monitor)
      if args.model_file is not None:
        agent.qnet.load_model_weights(args.model_file)
        _, avg, stddev = agent.test(100, agent.epsilon_target)
        print((avg, stddev))
      else:
        agent.train(100000)
    '''

    agent = DQN_Agent('MountainCar', deep=2)
    agent.qnet.load_model_weights('BestDueling')
    _, avg, stddev = agent.test(100, agent.epsilon_target)
    print("avg + std")
    print((avg, stddev))


if __name__ == '__main__':
    main(sys.argv)

