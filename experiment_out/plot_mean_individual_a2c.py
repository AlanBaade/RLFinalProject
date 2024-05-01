import csv
import plotter



fn = "experiment_out/a2c/soccer-individual-marl/progress.csv"
n_max = 1000


y = []
with open(fn, newline='') as csvfile:
  rdr = csv.reader(csvfile, delimiter=',')
  for i, row in enumerate(rdr):
    if 0<i and i<n_max:
      try:
        y.append(float(row[9]))
      except ValueError:
        pass
x = list(range(len(y)))
plotter.plot(x, y)


