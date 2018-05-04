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

  def sample_batch(self, batch_size=32):
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

# TODO (David): Implement importance sampling replay memory
# Idea - We can do importance sampling replay for memory in the RL Memory since
# it should be a good approximation of policy net anyway
# CBUFFER only for now

class IS_Replay_Memory():
  def __init__(self, game, agentsTypes, agent_index, memory_size=100000, burn_in=1000):

    self.cache = [None for _ in range(memory_size)]
    self.full_state_cache = [None for _ in range(memory_size)]
    self.full_action_cache = [None for _ in range(memory_size)]
    self.initial_likelihood = [0.0 for _ in range(memory_size)]
    self.cur_likelihood = [0.0 for _ in range(memory_size)]

    self.sample_prob = np.array([0.0 for _ in range(memory_size)])

    self.size = 0
    self.new = 0
    self.burn_in = burn_in
    self.cap = memory_size
    self.kind = CBUFFER

    self.agent_index = agent_index
    self.agentsTypes = agentsTypes


  # Calculates the probability that opponents of an agent would take the actionset
  # using the current policy networks
  def opponent_likelihood(self, stateset, actionset):
    agentsTypes = self.agentsTypes
    prob = 1.0
    for i in range(len(actionset)):
      if i != self.agent_index:
        policynet = agentsTypes[i].policynet
        agent_likelihood = policynet.action_prob(np.array([stateset[i]]), [actionset[i]])[0]
        prob *= agent_likelihood
    return prob

  def sample_batch(self, batch_size=32):
    # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
    # You will feed this to your model to train.

    max_index = min(self.size, self.cap)
    self.sample_prob = self.sample_prob / self.sample_prob.sum()
    batch_indices = np.random.choice(max_index, batch_size, p=self.sample_prob[:max_index])
    for i in batch_indices:
      stateset = self.full_state_cache[i]
      actionset = self.full_action_cache[i]
      self.cur_likelihood[i] = self.opponent_likelihood(stateset, actionset)
      self.sample_prob[i] = self.cur_likelihood[i] / self.initial_likelihood[i]
    return [self.cache[i] for i in batch_indices], [self.sample_prob[i] for i in batch_indices]

  def append(self, transition):
    if self.kind == CBUFFER:
      # Use circular buffer sampling
      self.cache[self.new] = transition
      self.full_action_cache[self.new] = transition[-2]
      self.full_state_cache[self.new] = transition[-1]
      self.initial_likelihood[self.new] = self.opponent_likelihood(transition[-1], transition[-2])
      self.cur_likelihood[self.new] = self.initial_likelihood[self.new]
      self.sample_prob[self.new] = self.initial_likelihood[self.new] / self.cur_likelihood[self.new]

      self.size = min(self.size + 1, self.cap)
      self.new = (self.new + 1) % self.cap

class Prioritized_Replay_Memory():

  def __init__(self, game, memory_size=100000, burn_in=1000, kind=CBUFFER):
    self.cache = [None for i in range(memory_size)]
    self.size = 0
    self.new = 0
    self.burn_in = burn_in
    self.cap = memory_size
    self.kind = kind
    self.max_priority = 1
    self.weights = [0 for i in range(memory_size)]

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