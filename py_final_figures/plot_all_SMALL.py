import util
import matplotlib.pyplot as plt


fig, ax = plt.subplots((1))


x, y = util.extract_xy("experiment_out/soccer-individual-marl/progress.csv", 1000)
util.plot(ax, x, y, "INDIVIDUAL")


x, y = util.extract_xy("experiment_out/soccer-joint-marl/progress.csv", 1000)
util.plot(ax, x, y, "JOINT")


x, y = util.extract_xy("experiment_out/soccer-joint-transformer/progress.csv", 1000)
util.plot(ax, x, y, "TRANSFORMER")


x, y = util.extract_xy("experiment_out/soccer-joint-linear-2/progress.csv", 1000)
util.plot(ax, x, y, "LINEAR")





ax.legend()
plt.xlim(0, 1200)
plt.show()

