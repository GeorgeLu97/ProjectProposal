# This game is setup to be a relatively simple game that for which the
# best strategy nontrivially depends on the strategies of the other agents

class MediumGame():
  ABSTAIN = 5

  def __init__(self):
    self.player_count = 5
    self.mafia_count = 2
    self.permutation = np.random.permutation(self.player_count)
    self.mafia = self.permutation[:self.mafia_count]
    self.is_mafia = [0 for _ in range(self.player_count)]
    for i in self.mafia:
      self.is_mafia[i] = 1

    self.alive = [1 for _ in range(self.player_count)]
    self.kill_matrix = [[0 for _ in range(self.player_count)] for _ in range(self.player_count)]

    # is_mafia, alive, kill_matrix
    self.state_size = self.player_count
    self.action_size = self.player_count + 1

    self.round_num = 0

  def get_villagers_alive(self):
    roles = self.is_mafia
    alive = self.alive
    for i in range(len(roles)):
      if roles[i] == 0 and alive[i] == 1:
        return True
    return False

  def get_mafia_alive(self):
    roles = self.is_mafia
    alive = self.alive
    for i in range(len(roles)):
      if roles[i] == 1 and alive[i] == 1:
        return True
    return False

  def step(self, actionset): 
    self.round_num += 1
    mark_dead = []
    for i in range(len(actionset)):
      if self.alive[i] == 1:
        target = actionset[i]
        if target == MediumMafia.ABSTAIN:
          continue
        actual_target = self.permutation_matrix[i][target]
        if random.random() < 0.33 and self.alive[actual_target] == 1:
          mark_dead.append(actual_target)
    for dead in mark_dead:
      self.alive[dead] = 0

    terminal = False
    reward = [0 for i in range(self.player_count)]
    if not self.get_villagers_alive() and not self.get_mafia_alive():
      terminal = True
    elif self.round_num >= 20:
      terminal = True
    elif not self.get_villagers_alive():
      reward = [1 if self.is_mafia[i] else -1 for i in range(self.player_count)]
      # reward = [1 if self.alive[i] else 0 for i in range(self.player_count)]
      terminal = True
    elif not self.get_mafia_alive():
      reward = [-1 if self.is_mafia[i] else 1 for i in range(self.player_count)]
      # reward = [1 if self.alive[i] else 0 for i in range(self.player_count)]
      terminal = True

    new_state = [self.get_state(i) for i in range(self.player_count)]

    return new_state, reward, terminal

  # If villager, self is always first, others are in an arbitrary permutation
  # If mafia, self is always first. Mafia is second,
  # others are in arbitrary permutation
  def get_state(self, agent):
    """
    own_position_portion = [1 if agent == i else 0 for i in range(self.player_count)]
    is_mafia_portion = self.is_mafia if self.is_mafia[agent] else [0 for _ in range(self.player_count)]
    """
    agent_permutation = self.permutation_matrix[agent]

    permuted_alive_matrix = [self.alive[i] for i in agent_permutation]
    permuted_kill_matrix = [[self.kill_matrix[i][j] for j in agent_permutation] for i in agent_permutation]
    # flat_kill_matrix = sum(permuted_kill_matrix, [])
    flat_kill_matrix = []
    state = permuted_alive_matrix + flat_kill_matrix

    state = np.array(state)
    return state

  def get_meta_state(self):
    return self.permutation_matrix

  def generate_player_permutation(self):
    permutation_matrix = []

    # Make permutation matrix for the mafia
    for i in range(self.mafia_count):
      agent_permutation = []
      agent_permutation.append(i) # self is first

      # Add all other mafia next
      for k in range(self.player_count):
        agent_num = self.permutation[k]
        if agent_num < self.mafia_count and agent_num != i:
          agent_permutation.append(agent_num)

      # Add all remaining players
      for k in range(self.player_count):
        agent_num = self.permutation[k]
        if agent_num >= self.mafia_count:
          agent_permutation.append(agent_num)
      permutation_matrix.append(agent_permutation)

    # Make permutation matrix for non-mafia:
    for i in range(self.mafia_count, self.player_count):
      agent_permutation = []
      agent_permutation.append(i)

      for k in range(self.player_count):
        agent_num = self.permutation[k]
        if agent_num != i:
          agent_permutation.append(agent_num)
      permutation_matrix.append(agent_permutation)

    return permutation_matrix

  def get_success_metrics(self):
    # Things that we currently think are good
    # Attempt to kill someone other than self
    # Killing Alive People to Dead people Ratio
    # Killing people who are trying to kill you

    return []

  def reset(self):
    # This is used to generate the permutation matrix
    self.permutation = np.random.permutation(self.player_count)

    # PermutationMatrix[i][j] = who the jth person should be the ith player
    self.permutation_matrix = self.generate_player_permutation()

    self.is_mafia = [0 for _ in range(self.player_count)]
    for i in range(self.mafia_count):
      self.is_mafia[i] = 1

    # These are the objective alive and kill matrices
    self.alive = [1 for _ in range(self.player_count)]
    self.kill_matrix = [[0 for _ in range(self.player_count)] for _ in range(self.player_count)]

    self.round_num = 0

    return [self.get_state(i) for i in range(self.player_count)]