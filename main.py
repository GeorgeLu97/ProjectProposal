#!/usr/bin/env python
### TODO

import matplotlib as mpl
import keras
import numpy as np, gym, sys, copy, argparse
import tensorflow as tf

import random
import time
import games
import rps
import mediumgames
import math

from parameters import parameters_dict
from networks import PNetwork, QNetwork
import replay
from replay import Replay_Memory

from utils import *

class DQN_Agent():

  # In this class, we will implement functions to do the following.
  # (1) Create an instance of the Q Network class.
  # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
  #   (a) Epsilon Greedy Policy.
  #     (b) Greedy Policy.
  # (3) Create a function to train the Q Network, by interacting with the environment.
  # (4) Create a function to test the Q Network's performance on the environment.
  # (5) Create a function for Experience Replay.

  def __init__(self, game, parameters=None, render=False, use_replay=False,
         deep=0, monitor=False):

    # Create an instance of the network itself, as well as the memory.
    # Here is also a good place to set environmental parameters,
    # as well as training parameters - number of episodes / iterations, etc.

    self.gamma = 0.99
    self.RLalpha = 0.1
    self.SLalpha = 0.005

    self.RLBufferSize = 1000
    self.SLBufferSize = 50000

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


    self.target_update_period = 100

    self.network_update_period = 128
    self.network_updates = 2

    self.iteration = 0

    self.brp = True
    self.sigma = self.brp_action

    self.replayRL = Replay_Memory(game, memory_size=self.RLBufferSize,
       kind=replay.CBUFFER)
    self.replaySL = Replay_Memory(game, memory_size=self.SLBufferSize,
       kind=replay.RESERVOIR)

  # q_values: State * Action -> Value
  def brp_action(self, state):
    if random.random() < self.epsilon:
      action = random.randint(0, self.action_size - 1)
      return action
    else:
      best_action, _ = self.valuenet.best_action(state)
      return best_action

  # greedy policy
  def average_policy_action(self, state):
    best_action = self.policynet.best_action(state)
    return best_action

  def resetepisode(self, average_only=False):
    if not average_only and random.random() < self.eta:
      self.brp = True
      self.sigma = self.brp_action
    else:
      self.brp = False
      self.sigma = self.average_policy_action

  def act(self, state):
    action = self.sigma(state)
    return action

  def updatereplay(self, state, action, reward, next_state, done):
    # See paper for recommended epsilon decay
    self.epsilon = self.epsilon_initial / math.ceil(math.sqrt((self.iteration + 1)  / 10000))

    self.iteration += 1
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

  # Not Essential
  def surveySLMemory(self):
    freqs = [0 for i in range(self.action_size)]
    for item in self.replaySL.cache[:self.replaySL.size]:
      for j in range(self.action_size):
        if item[1][j] == 1:
          freqs[j] += 1

    print(freqs)

  # Not Essential
  def surveyRLMemory(self):
    for item in self.replayRL.cache[:self.replayRL.size]:
      print(item)


class RandomAgent():
  def __init__(self, env):
    self.action_size = env.action_size

  def resetepisode(self, training=False):
    pass

  def act(self, state):
    return random.randint(0, self.action_size - 1)

class DQN_Game():

  def __init__(self, environment_name, render=False, use_replay=False,
         deep=0, monitor=False):

    self.env = rps.RPS() # current environment

    self.action_size = self.env.action_size

    self.agents = [i for i in range(self.env.player_count)]
    self.num_agents = len(self.agents)

    self.parameters = parameters_dict[environment_name]

    # For now I'm enforcing one agent per agentType. Maybe this can change in the future
    self.agentsTypes = [DQN_Agent(self, parameters=self.parameters)
        for _ in range(len(self.agents))]
    self.num_agent_types = len(self.agentsTypes)

    self.render = render
    self.use_replay = use_replay

    # Burns in memory for all agents
    self.burn_in_memory(self.agentsTypes[0].replayRL.burn_in)

    print("Completed All Agent and Game Initialization")

  def train(self, episodes):
    testing_rewards = []
    testing_team_rewards = []
    exploitabilities = []
    run_test = False

    iteration = 0
    freqs = [[0 for _ in range(self.action_size)] for _ in range(self.num_agent_types)]
    for episode in range(episodes):
      [i.resetepisode() for i in self.agentsTypes]
      cur_state = self.env.reset()
      while True:
        iteration += 1

        if iteration % 5000 == 0:
          run_test = True

        actionset = [self.agentsTypes[i].act(cur_state[i]) for i in range(len(self.agents))]
        next_state, rewards, is_terminal = self.env.step(actionset)
        for i in range(self.num_agent_types):
          if self.agentsTypes[i].brp:
            freqs[i][actionset[i]] += 1

        for i in range(self.num_agent_types):
          self.agentsTypes[i].updatereplay(cur_state[i], actionset[i],
             rewards[i], next_state[i], is_terminal)
          
        cur_state = next_state
        if is_terminal:
          break

      if run_test:
        print(freqs)
        freqs = [[0 for _ in range(self.action_size)] for _ in range(self.num_agent_types)]

        exploitabilities.append(self.check_exploitability())
        avg_score_differential, avg_team_scores = self.test()
        testing_rewards.append(avg_score_differential)
        testing_team_rewards.append(avg_team_scores)

        run_test = False


    print(testing_rewards)
    print(rotate_stats(testing_team_rewards))
    print(rotate_stats(exploitabilities))
    print("completed training")
    for agentType in self.agentsTypes:
      agentType.surveySLMemory()

  def check_exploitability(self):
    es = []
    for team in range(self.env.num_teams):
      agent = self.agentsTypes[team]
      print(agent.valuenet.predict(np.array([[0]]))[0])
      e = self.env.compute_exploitability(team, agent.policynet)
      es.append(e)
    print(es)
    return es

  # For testing, we test set one team to be our trained agents and the other team to be random agents
  def test(self):
    NUM_TEST_ITERS = 1000

    total_ai_reward = 0
    ai_role_reward = [0 for _ in range(self.env.num_teams)]
    ai_role_count = [0 for _ in range(self.env.num_teams)]
    freqs = [[0 for _ in range(self.action_size)] for _ in range(self.num_agent_types)]

    for episode in range(NUM_TEST_ITERS):
      cur_state = self.env.reset()
      [i.resetepisode(average_only=True) for i in self.agentsTypes]

      team_list = self.env.team
      random_ai = random.randint(0, self.env.num_teams - 1)

      agent_list = [None for _ in range(len(self.agents))]

      for i in range(len(self.agents)):
        if team_list[i] == random_ai:
          agent_list[i] = RandomAgent(self.env)
        else:
          agent_list[i] = self.agentsTypes[i]

      while True:
        actionset = [agent_list[i].act(cur_state[i]) for i in range(len(agent_list))]
        next_state, rewards, is_terminal = self.env.step(actionset)
        for i in range(self.num_agent_types):
          if team_list[i] != random_ai:
            freqs[i][actionset[i]] += 1

        cur_state = next_state
        if is_terminal:
          for i in range(self.num_agents):
            # If this agent is not controlled by the random ai
            if team_list[i] != random_ai:
              total_ai_reward += rewards[i]
              ai_role_count[team_list[i]] += 1
              ai_role_reward[team_list[i]] += rewards[i]
              break
          break
    print(freqs)
    ai_average_reward = [(ai_role_reward[i] / ai_role_count[i]) for i in range(self.num_agent_types)]

    return total_ai_reward / NUM_TEST_ITERS, ai_average_reward


  def burn_in_memory(self, bns):
    cur_state = self.env.reset()
    # Initialize your replay memory with a burn_in number of episodes / transitions.
    for _ in range(0, bns):
      actionset = [self.agentsTypes[i].act(cur_state[i]) for i in range(len(self.agents))]
      next_state, rewards, is_terminal = self.env.step(actionset)

      [self.agentsTypes[i].appendreplay(cur_state[i], actionset[i], rewards[i], next_state[i], is_terminal)
       for i in range(len(self.agents))]

      cur_state = next_state
      if is_terminal:
        cur_state = self.env.reset()


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

  # args = parse_arguments()
  # environment_name = args.env

  game = DQN_Game('rps')
  game.train(500000)


if __name__ == '__main__':
  main(sys.argv)

