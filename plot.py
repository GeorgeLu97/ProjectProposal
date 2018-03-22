import matplotlib.pyplot as plt
import numpy as np


results = []
x = range(len(results))

fig = plt.figure()

plt.scatter(x, results, s=10, label='Average reward against random agent')
plt.legend(loc='best')
plt.show()
plt.savefig('randomagent.png')