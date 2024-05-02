import util
import matplotlib.pyplot as plt


fig, ax = plt.subplots((1))


x, y = util.extract_xy("experiment_out/soccer-individual-marl/progress.csv", 1000)
util.plot(ax, x, y, "INDIVIDUAL")


x, y = util.extract_xy("experiment_out/soccer-joint-marl/progress.csv", 1000)
util.plot(ax, x, y, "JOINT")


x, y = util.extract_xy("experiment_out/agrew/soccer-individual-marl/progress.csv", 1000)
util.plot(ax, x, y, "INDIVIDUAL AGENT REWARD")


x, y = util.extract_xy("experiment_out/agrew/soccer-joint-marl/progress.csv", 1000)
util.plot(ax, x, y, "JOINT AGENT REWARD")





ax.legend()
plt.xlim(0, 1200)
plt.show()

