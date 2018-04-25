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
import math

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
    model.add(Dense(30, activation='relu', input_dim=(self.state_size)))
    #model.add(Dense(30, activation='relu'))
    #model.add(Dense(30, activation='relu'))
    model.add(Dense(self.action_size, activation='relu'))

    #adam = optimizers.Adam(lr=self.agent.alpha, decay=1e-6)
    sgd = optimizers.SGD(lr=self.agent.RLalpha)
    model.compile(loss='mse',
            optimizer=sgd,
            metrics=['accuracy'])

    model2 = Sequential()
    model2.add(Dense(30, activation='relu', input_dim=(self.state_size)))
    #model2.add(Dense(30, activation='relu'))
    #model2.add(Dense(30, activation='relu'))
    model2.add(Dense(self.action_size, activation='relu'))

    # adam = optimizers.Adam(lr=self.agent.alpha, decay=1e-6)
    sgd = optimizers.SGD(lr=self.agent.RLalpha)
    model2.compile(loss='mse',
            optimizer=sgd,
            metrics=['accuracy'])

    self.modeltrain = model
    self.modeltarget = model2


  # uses target
  def predict(self, states, model=None):
    if model is None:
      model = self.modeltarget
    return model.predict(states)

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
    cur_targets = self.predict(states, self.modeltrain)

    # Basically sets cur_targets[actions[i]] = new_targets[i] for each i
    cur_targets[np.arange(cur_targets.shape[0]), actions] = new_targets

    self.modeltrain.fit(states, cur_targets, batch_size=size, verbose=0)

  def update(self, state, action, reward, next_state, is_terminal):
    self.update_batch(1, [[state, action, reward, next_state, is_terminal]])

  def update_target(self):
    self.modeltarget.set_weights(self.modeltrain.get_weights())

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
    model.add(Dense(30, activation='relu', input_dim=(self.state_size)))
    #model.add(Dense(30, activation='relu'))
    #model.add(Dense(30, activation='relu'))
    model.add(Dense(self.action_size, activation='softmax'))

    #adam = optimizers.Adam(lr=self.agent.alpha, decay=1e-6)
    sgd = optimizers.SGD(lr=self.agent.SLalpha)
    model.compile(loss='categorical_crossentropy',
            optimizer=sgd,
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
    batch = self.predict(states)

    random_actions = []
    for probs in batch:
      random_actions.append(np.random.choice(self.action_size, 1, p=probs)[0])
    return random_actions

  def best_action(self, state):
    return self.best_action_batch(np.array([state]))[0]

  def update_batch(self, size, experienceList):
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

  def __init__(self, game, memory_size=100000, burn_in=1000, kind = CBUFFER):
    self.cache = [None for i in range(memory_size)]
    self.size = 0
    self.new = 0
    self.burn_in = burn_in
    self.cap = memory_size
    self.kind = kind

    # The memory essentially stores transitions recorder from the agent
    # taking actions in the environment.
    # Burn in episodes define the number of episodes that are written into the memory from the
    # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
    # A simple (if not the most efficient) was to implement the memory is as a list of transitions.


  def sample_batch(self, batch_size=32):
    # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
    # You will feed this to your model to train.
    return random.sample(self.cache[:min(self.size, self.cap)], batch_size)

  def append(self, transition):
    if self.kind == CBUFFER:
      # Use circular buffer sampling
      self.cache[self.new] = transition
      self.size = min(self.size + 1, self.cap)
      self.new = (self.new + 1) % self.cap
    else:
      # Use Reservoir Sampling
      if self.size < self.cap:
        self.cache[self.size] = transition
        self.size += 1
      else:
        if random.random() * self.size < self.cap:
          self.cache[random.randint(0, self.cap - 1)] = transition
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

    self.gamma = 0.99
    self.RLalpha = 0.1
    self.SLalpha = 0.005

    self.epsilon_initial = 0.5
    self.epsilon = self.epsilon_initial

    self.episodes = 1000000
    self.env = game.env
    self.state_size = self.env.state_size
    self.action_size = self.env.action_size
    self.eta = 0.1

    self.deep = deep

    self.policynet = PNetwork(self.env, self, deep=deep)
    self.valuenet = QNetwork(self.env, self, deep=deep)

    self.target_update_period = 1000
    self.network_update_period = 128
    self.network_updates = 2

    self.iteration = 0

    self.brp = True
    self.sigma = self.epsilon_greedy_policy

  def init_replay(self, game):
    self.replayRL = Replay_Memory(game, kind=CBUFFER)
    self.replaySL = Replay_Memory(game, kind=RESERVOIR)

  # q_values: State * Action -> Value
  def epsilon_greedy_policy(self, state):
    if random.random() < self.epsilon:
      action = random.randint(0, self.action_size - 1)
      return action
    else:
      best_action, _ = self.valuenet.best_action(state)
      return best_action

  # greedy policy
  def greedy_policy(self, state):
    best_action = self.policynet.best_action(state)
    return best_action

  def resetepisode(self, testing=False):
    if testing:
      self.brp = False
      self.sigma = self.greedy_policy
    else:
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
    # See paper for recommended epsilon decay
    self.epsilon = self.epsilon_initial / math.ceil(math.sqrt((self.iteration + 1)  / 10000))

    self.iteration += 1
    self.state = next_state
    self.replayRL.append([state, action, reward, next_state, done])
    if self.brp:
      action_onehot = [0 for _ in range(self.action_size)]
      action_onehot[action] = 1
      self.replaySL.append([state, action_onehot])

    if self.iteration % self.network_update_period == 0:
      for i in range(self.network_updates):
        batch = self.network_update_period

        replayRLbatch = self.replayRL.sample_batch(batch)
        # should use mse loss, just have # action_size results
        self.valuenet.update_batch(batch, replayRLbatch)

        if self.replaySL.size >= 1000:
          replaySLbatch = self.replaySL.sample_batch(batch)
          # should use crossentropy loss, softmax activation
          self.policynet.update_batch(batch, replaySLbatch)

    if self.iteration % self.target_update_period == 0:
      self.valuenet.update_target()

  def appendreplay(self, state, action, reward, next_state, done):
    self.replayRL.append([state, action, reward, next_state, done])

  def surveySLMemory(self):
    freqs = [0 for i in range(self.action_size)]
    for item in self.replaySL.cache[:self.replaySL.size]:
      for j in range(self.action_size):
        if item[1][j] == 1:
          freqs[j] += 1

    print(freqs)

class RandomAgent():
  def resetepisode(self, training=False):
    pass

  def act(self, state):
    return random.randint(0, 5)

class DQN_Game():

  def __init__(self, environment_name, render=False, use_replay=False,
         deep=0, monitor=False):

    self.env = games.ToyEnvironment() # current environment

    #self.agents = [DQN_Agent(self) for i in range(self.env.player_count)]

    self.agents = [i for i in range(self.env.player_count)]

    self.agentsTypes = [DQN_Agent(self) for i in range(2)]

    self.render = render
    self.use_replay = use_replay
    # [i.init_replay(self) for i in self.agents]
    [i.init_replay(self) for i in self.agentsTypes]

    self.burn_in_memory(self.agentsTypes[0].replayRL.burn_in)
    print("Completed All Agent and Game Initialization")

  def train(self, episodes=5000):
    # Unused right now
    testing_rewards = []
    townie_rewards = []
    mafia_rewards = []

    run_test = False

    cur_state = self.env.reset()
    meta_state = self.env.get_meta_state()

    iteration = 0
    freqs = [0 for _ in range(len(self.agents) + 1)]
    for episode in range(episodes):
      [i.resetepisode() for i in self.agentsTypes]
      prevstates = self.env.reset()
      while True:
        iteration += 1

        if iteration % 5000 == 0:
          run_test = True

        actionset = [self.agentsTypes[self.env.is_mafia[i]].act(cur_state[i]) for i in range(len(self.agents))]
        next_state, rewards, is_terminal = self.env.step(actionset)
        if self.agentsTypes[0].brp:
          freqs[actionset[0]] += 1

        for i in range(len(self.agents)):
          self.agentsTypes[self.env.is_mafia[i]].updatereplay(cur_state[i], actionset[i],
             rewards[i], next_state[i], is_terminal)
          
        cur_state = next_state
        if is_terminal:
          break

      if run_test:
        print(freqs)
        freqs = [0 for _ in range(len(self.agents) + 1)]

        avg_score_differential, avg_score_townie, avg_score_mafia = self.test()
        testing_rewards.append(avg_score_differential)
        townie_rewards.append(avg_score_townie)
        mafia_rewards.append(avg_score_mafia)
        run_test = False

    print(testing_rewards)
    print(townie_rewards)
    print(mafia_rewards)
    print("completed training")

  # For testing, we test set one team to be our trained agents and the other team to be random agents
  def test(self):
    total_ai_reward = 0
    ai_role_reward = [0, 0]
    ai_role_count = [0, 0]
    freqs = [0 for _ in range(len(self.agents) + 1)]

    for episode in range(500):
      cur_state = self.env.reset()
      [i.resetepisode(testing=True) for i in self.agentsTypes]
      mafia_list = self.env.is_mafia
      random_ai = random.randint(0, 1)
      agent_list = [None for _ in range(len(self.agents))]

      for i in range(len(self.agents)):
        if mafia_list[i] == random_ai:
          agent_list[i] = RandomAgent()
        else:
          agent_list[i] = self.agentsTypes[0] #0 is town

      while True:
        actionset = [agent_list[i].act(cur_state[i]) for i in range(len(agent_list))]
        next_state, rewards, is_terminal = self.env.step(actionset)
        if mafia_list[0] != random_ai:
          freqs[actionset[0]] += 1

        cur_state = next_state
        if is_terminal:
          for i in range(len(self.agents)):
            # If this agent is not controlled by the random ai
            if mafia_list[i] != random_ai:
              total_ai_reward += rewards[i]
              ai_role_count[1 - random_ai] += 1
              ai_role_reward[1 - random_ai] += rewards[i]
              break
          break
    print(freqs)
    print("Townie: " + str(ai_role_reward[0] / ai_role_count[0]))
    print("Mafia: " + str(ai_role_reward[1] / ai_role_count[1]))
    return (total_ai_reward / 500, ai_role_reward[0] / ai_role_count[0], ai_role_reward[1] / ai_role_count[1])


  def burn_in_memory(self, bns):
    cur_state = self.env.reset()
    # Initialize your replay memory with a burn_in number of episodes / transitions.
    for _ in range(0, bns):
      actionset = [self.agentsTypes[self.env.is_mafia[i]].act(cur_state[i]) for i in range(len(self.agents))]
      next_state, rewards, is_terminal = self.env.step(actionset)

      [self.agentsTypes[self.env.is_mafia[i]].appendreplay(cur_state[i], actionset[i], rewards[i], next_state[i], is_terminal)
       for i in range(len(self.agents))]

      cur_state = next_state
      if is_terminal:
        cur_state = self.env.reset()

    # need to episode to finish or else monitor complains
    while True:
      actionset = [self.agentsTypes[self.env.is_mafia[i]].act(cur_state[i]) for i in range(len(self.agents))]
      next_state, rewards, is_terminal = self.env.step(actionset)

      [self.agentsTypes[self.env.is_mafia[i]].appendreplay(cur_state[i], actionset[i], rewards[i], next_state[i], is_terminal)
       for i in range(len(self.agents))]

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


  game = DQN_Game('MountainCar')
  game.train(500000)


if __name__ == '__main__':
  main(sys.argv)

