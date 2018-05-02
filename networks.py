from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import optimizers
from keras.engine.topology import Layer

import numpy as np, gym, sys, copy, argparse

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
    model.add(Dense(self.action_size, activation='sigmoid'))

    #adam = optimizers.Adam(lr=self.agent.alpha, decay=1e-6)
    sgd = optimizers.SGD(lr=self.agent.RLalpha)
    model.compile(loss='mse',
            optimizer=sgd,
            metrics=['accuracy'])

    model2 = Sequential()
    model2.add(Dense(30, activation='relu', input_dim=(self.state_size)))
    #model2.add(Dense(30, activation='relu'))
    #model2.add(Dense(30, activation='relu'))
    model2.add(Dense(self.action_size, activation='sigmoid'))
    # Note on Sigmoid vs ReLU
    # reLU is bad for our RPS example since all close to 0 output is hard for ReLU to
    # model well

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
    #if random.random() < 0.001:
    #  print(next_state_actions)
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
