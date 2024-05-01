import matplotlib.pyplot as plt
import csv



def extract_xy(fn, n_max):
  x = []
  y = []
  with open(fn, newline='') as csvfile:
    rdr = csv.reader(csvfile, delimiter=',')
    col1 = -1
    col2 = -1
    rdr = list(rdr)
    for i, row in enumerate(rdr):
      if 0==i:
        col1 = row.index("train/n_updates")
        col2 = row.index("rollout/ep_rew_mean")
      if 0<i and i<n_max:
        try:
          x.append(float(row[col1]))
          y.append(float(row[col2]))
        except ValueError:
          pass
  return (x, y)



def preprocess(y):
  lam = 0
  y2 = [y[0]]
  for i in range(len(y)-1):
    y2.append(lam * y2[i-1] + (1-lam) * y[i])
  return y2



def plot(surface, x, y, title):
  y = preprocess(y)
  surface.plot(x, y, label=title)
  surface.set_xlabel('# Updates')
  surface.set_ylabel('Mean Reward')
  surface.set_title(title)



