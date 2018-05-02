""" This might be useful eventually when we have many environments """

parameters_dict = {}
rps_parameters = { 
  'eta': 0.5,
  'gamma': 0.99,
  'RLalpha': 0.1,
  'SLalpha': 0.005,
  'RLBufferSize': 1000,
  'SLBufferSize': 50000
}

parameters_dict['rps'] =rps_parameters