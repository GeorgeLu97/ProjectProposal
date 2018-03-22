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
import games

DUEL = 2
DEEP = 1

RESERVOIR = 0
CBUFFER = 1

class QNetwork():

    def __init__(self, env, agent, deep=0):

        self.modeltrain = None
        self.modeltarget = None
        self.agent = agent

        self.state_size = env.state_size
        self.action_size = env.action_size


        model = Sequential()
        model.add(Dense(10, activation='relu', input_dim=(self.state_size)))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))

        adam = optimizers.Adam(lr=self.agent.alpha, decay=1e-6)
        model.compile(loss='mse',
                      optimizer=adam,
                      metrics=['accuracy'])

        model2 = Sequential()
        model2.add(Dense(10, activation='relu', input_dim=(self.state_size)))
        model2.add(Dense(10, activation='relu'))
        model2.add(Dense(10, activation='relu'))
        model2.add(Dense(self.action_size, activation='relu'))

        adam = optimizers.Adam(lr=self.agent.alpha, decay=1e-6)
        model2.compile(loss='mse',
                      optimizer=adam,
                      metrics=['accuracy'])

        self.modeltrain = model
        self.modeltarget = model2


    # uses train
    def predict(self, states):
        return self.modeltarget.predict(states)

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

        self.modeltrain.fit(states, cur_targets, batch_size=size, verbose=0)

    def update(self, state, action, reward, next_state, is_terminal):
        self.update_batch(1, [[state, action, reward, next_state, is_terminal]])

    def update_target(self):
        self.modeltarget.set_weights(self.modeltrain.get_weights)

    def save_model_weights(self, weight_file1, weight_file2):
        # Helper function to save your model / weights.
        self.modeltrain.save_weights(weight_file1)
        self.modeltarget.save_weights(weight_file2)
        pass

    def load_model(self, model_file1, model_file2):
        # Helper function to load an existing model.
        self.modeltrain = keras.models.load_model(model_file1)
        self.modeltarget = keras.models.load_model(model_file2)

    def load_model_weights(self, weight_file1, weight_file2):
        # Helper function to load model weights.
        self.modeltrain.load_weights(weight_file1)
        self.modeltarget.load_weights(weight_file2)

class PNetwork():

    def __init__(self, env, agent, deep=0):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.

        self.model = None
        self.agent = agent

        self.state_size = env.state_size
        self.action_size = env.action_size

        model = Sequential()
        model.add(Dense(10, activation='relu', input_dim=(self.state_size)))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))
        model.add(Dense(1, activation='softmax'))

        adam = optimizers.Adam(lr=self.agent.alpha, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])
        self.model = model

    # states : np.array (num_inputs, num_dims)
    # return : np.array (num_inputs, num_actions)
    def predict(self, states):
        return self.model.predict(states)

    # states : np.array (num_inputs, num_dims)
    # return best_action : (num_inputs, )
    # return best_action_value : (num_inputs, )
    def best_action_batch(self, states):
        return self.predict(states)

    def best_action(self, state):
        return self.best_action_batch(np.array([state]))[0]

    def update_batch(self, size, experienceList):
        target_list = []
        cur_list = []

        states = np.array([experience[0] for experience in experienceList])
        actions = np.array([experience[1] for experience in experienceList])

        self.model.fit(states, actions, batch_size=size, verbose=0)

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

    def __init__(self, game, memory_size=50000, burn_in=10000, kind = CBUFFER):
        self.cache = [None for i in range(memory_size)]
        self.size = burn_in
        self.new = burn_in
        self.cap = memory_size
        self.kind = kind

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.
        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

        # initialize burn in
        game.burn_in_memory(self.cache, burn_in)

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        return random.sample(self.cache[:self.size], batch_size)

    def append(self, transition):
        if self.kind == CBUFFER:
            self.cache[self.new] = transition
            self.size = min(self.size + 1, self.cap)
            self.new = (self.new + 1) % self.cap
        else:
            if self.size < self.cap:
                self.cache[self.size] = transition
                self.size += 1
            else:
                if random.random() * self.size < self.cap:
                    self.cache[random.randint(0, self.cap-1)] = transition
                    self.size += 1
                else:
                    self.size += 1




class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #   (a) Epsilon Greedy Policy.
    #     (b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, game, render=False, use_replay=False,
                 deep=0, monitor=False):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.

        self.alpha = 0.0001
        self.epsilon = 0.5
        self.epsilon_target = 0.05
        self.epsilon_delta = (0.5 - 0.05) / 100000
        self.episodes = 1000000
        self.env = game.env
        self.eta = 0.5

        self.deep = deep

        self.policynet = PNetwork(self.env, self, deep=deep)
        self.valuenet = QNetwork(self.env, self, deep = deep)

        # make targetnet just part of valuenet
        self.targetnet = self.valuenet # yeah i know this isn't what i want

        self.update_period = 1000

        self.iteration = 0
        if random.random() < self.eta:
            self.brp = True
            self.sigma = self.epsilon_greedy_policy
        else:
            self.brp = False
            self.sigma = self.greedy_policy

    def init_replay(self, game):
        self.replayRL = Replay_Memory(game, kind=CBUFFER)
        self.replaySL = Replay_Memory(game, kind=RESERVOIR)

    # q_values: State * Action -> Value
    def epsilon_greedy_policy(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            best_action, _ = self.valuenet.best_action(state)
            return best_action

    # greedy policy
    def greedy_policy(self, state):
        best_action = self.policynet.best_action(state)
        return best_action

    def resetepisode(self):
        self.iteration = 0
        if random.random() < self.eta:
            self.brp = True
            self.sigma = self.epsilon_greedy_policy
        else:
            self.brp = False
            self.sigma = self.greedy_policy

    def act(self):
        action = self.sigma(self.state)
        return action

    def act(self, state):
        action = self.sigma(state)
        return action

    def updatereplay(self, state, action, reward, next_state, done):
        self.iteration += 1
        self.state = next_state
        self.replayRL.append([state, action, reward, next_state, done])
        if self.brp:
            self.replaySL.append([state, action])

        replayRLbatch = self.replayRL.sample(32)
        replaySLbatch = self.replaySL.sample(32)

        #should use crossentropy loss, softmax activation
        self.policynet.update_batch(32, replaySLbatch)

        #should use mse loss, just have # action_size results
        self.valuenet.update_batch(32, replayRLbatch)
        if self.iteration % self.update_period == 0:
            self.valuenet.update_target()

    def appendreplay(self, state, action, reward, next_state, done):
        self.replayRL.append([state, action, reward, next_state, done])
        if self.brp:
            self.replaySL.append([state, action])


class DQN_Game():

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

        self.env = games.ToyEnvironment() # current environment

        # true/false input to whether good or bad agent
        self.agents = [DQN_Agent(self) for i in range(self.env.player_count)]

        self.render = render
        self.use_replay = use_replay
        [i.init_replay(self) for i in self.agents]

    def train(self, episodes=5000):
        testing_rewards = []

        run_test = False
        for episode in range(episodes):
            [i.resetepisode() for i in self.agents]
            iteration = 0
            prevstates = self.env.reset()

            while True:
                iteration += 1

                if iteration % 10000 == 0:
                    run_test = True

                actions = [i.act() for i in self.agents]
                results, reward, is_terminal = self.env.step1(actions)
                [self.agents[i].updatereplay(prevstates[i], actions[i], reward[i], results[i], is_terminal)
                    for i in range(len(self.agents))]
                prevstates = results
                if is_terminal:
                    break

    def burn_in_memory(self, cache, bns):
        cur_state = self.env.reset()
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        for i in range(0, bns):
            actionset = [self.agents[i].act(cur_state[i]) for i in range(len(self.agents))]
            next_state, rewards, is_terminal = self.env.step(actionset)

            [self.agents[i].appendreplay(cur_state[i], actionset[i], rewards[i], next_state[i], is_terminal)
             for i in range(len(self.agents))]

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


'''
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
'''

def main(args):
    '''
    args = parse_arguments()
    environment_name = args.env
    '''

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

    agent = DQN_Game('MountainCar')
    agent.train(1000)
    # _, avg, stddev = agent.test(100, agent.epsilon_target)
    #print("avg + std")
    # print((avg, stddev))


if __name__ == '__main__':
    main(sys.argv)

