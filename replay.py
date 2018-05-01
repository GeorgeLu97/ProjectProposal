import random
import numpy as np

RESERVOIR = 0
CBUFFER = 1

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

class Prioritized_Replay_Memory():
  RESERVOIR = 0
  CBUFFER = 1

  def __init__(self, game, memory_size=100000, burn_in=1000, kind=CBUFFER):
    self.cache = [None for i in range(memory_size)]
    self.size = 0
    self.new = 0
    self.burn_in = burn_in
    self.cap = memory_size
    self.kind = kind
    self.max_priority = 1
    self.weights = [0 for i in range(memory_size)]

    # The memory essentially stores transitions recorder from the agent
    # taking actions in the environment.
    # Burn in episodes define the number of episodes that are written into the memory from the
    # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
    # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

  def sample_batch(self, agent, batch_size=32):
    # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
    # You will feed this to your model to train.
    # print(np.random.choice(self.cache, batch_size, p=self.weights (but normalized)))
    true_weights = self.weights / np.sum(self.weights)
    indices = np.random.choice(list(range(min(self.size, self.cap))), batch_size, p=true_weights)
    # need to update weights/priority here too
    for i in indices:
      [state, action, reward, next_state, done] = self.cache[i]
      self.weights[i] = abs(reward + self.agent.gamma * np.max(agent.QNetwork.model.predict([next_state])[0]) - agent.QNetwork.model.predict([state])[0][action])

    return random.sample(self.cache[:min(self.size, self.cap)], batch_size)

  def append(self, transition):
    if self.kind == CBUFFER:
      # Use circular buffer sampling
      self.cache[self.new] = transition
      self.weights[self.new] = self.max_priority
      self.size = min(self.size + 1, self.cap)
      self.new = (self.new + 1) % self.cap
    else:
      # Use Reservoir Sampling
      if self.size < self.cap:
        self.cache[self.size] = transition
        self.weights[self.size] = self.max_priority
        self.size += 1
      else:
        if random.random() * self.size < self.cap:
          ind = random.randint(0, self.cap - 1)
          self.cache[ind] = transition
          self.weights[ind] = self.mex_priority
          self.size += 1
        else:
          self.size += 1