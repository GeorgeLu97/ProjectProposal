import numpy as np

class RPS():
  def __init__(self):
    self.player_count = 2
    self.num_teams = 2
    self.team = None # Will be filled in by game

    # Keep these up to date
    self.state_size = 1
    self.action_size = 3

    # Initialize the first game
    self.reset()

    # We will save some info about past games here
    # We might query this to see how well our agents are learning
    self.metrics = []

  def step(self, actionset): 
    terminal = False
    reward = [0, 0]

    terminal = True
    self.game_state = 1

    if actionset[0] == (actionset[1] + 1) % 3:
      reward = [self.rps_rewards[0][actionset[0]], -self.rps_rewards[0][actionset[0]]]
    elif actionset[1] == (actionset[0] + 1) % 3:
      reward = [-self.rps_rewards[1][actionset[1]], self.rps_rewards[1][actionset[1]]]
    else:
      reward = [0, 0]

    new_state = [self.get_state(i) for i in range(self.player_count)]

    return new_state, reward, terminal

  def get_state(self, agent):
    return [self.game_state]

  def get_meta_state(self):
    return self.game_state

  def get_success_metrics(self):
    return 

  def save_success_metrics(self):    
    return

  # Optimal exploit strategy is always a single action
  def compute_exploitability(self, team, pnetwork):
    action_probs = pnetwork.predict(np.array([[0]]))[0]
    # print(action_probs)
    max_exploitability = -100.0
    for action_taken in range(3):
      value = (self.rps_rewards[team][action_taken] * action_probs[(action_taken + 2) % 3]
          - self.rps_rewards[1 - team][(action_taken + 1) % 3] * action_probs[(action_taken + 1) % 3])
      max_exploitability = max(value, max_exploitability)

    return max_exploitability

  def reset(self, save_metrics=False):
    self.team = [0, 1] 
    self.rps_rewards = [[0.5, 1, 2], [0.5, 1, 2]]

    self.game_state = 0

    return [self.get_state(i) for i in range(self.player_count)]