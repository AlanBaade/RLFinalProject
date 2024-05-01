import matplotlib.pyplot as plt

def plot(x, y):
  plt.plot(x, y)
  plt.xlabel('steps')
  plt.ylabel('mean reward')
  plt.show()  

