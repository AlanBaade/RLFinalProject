import plotter
import csv

fn = "experiment_out/soccer-joint-transformer/progress.csv"
n_max = 1000


y = []
with open(fn, newline='') as csvfile:
  rdr = csv.reader(csvfile, delimiter=',')
  for i, row in enumerate(rdr):
    if 0<i and i<n_max:
      try:
        y.append(float(row[1]))
      except ValueError:
        pass
x = list(range(len(y)))
plotter.plot(x, y)


