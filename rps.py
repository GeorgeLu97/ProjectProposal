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

    if actionset[0] == (actionset[1] + 1) % 3:
      terminal = True
      self.game_state = 0
      reward = [self.rps_rewards[0][actionset[0]], -self.rps_rewards[0][actionset[0]]]
    elif actionset[1] == (actionset[0] + 1) % 3:
      terminal = True
      self.game_state = 0
      reward = [-self.rps_rewards[1][actionset[1]], self.rps_rewards[1][actionset[1]]]

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

  def reset(self, save_metrics=False):
    self.team = [0, 1] 
    self.rps_rewards = [[1, 1, 1], [1, 1, 1]]

    self.game_state = 1

    return [self.get_state(i) for i in range(self.player_count)]