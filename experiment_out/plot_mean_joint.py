import matplotlib.pyplot as plt
import csv

fn = "experiment_out/soccer-joint-marl/progress.csv"

y = []
with open(fn, newline='') as csvfile:
  rdr = csv.reader(csvfile, delimiter=',')
  for i, row in enumerate(rdr):
    if i > 0:
      y.append(float(row[3]))
x = list(range(len(y)))
plt.plot(x, y)
plt.show()


