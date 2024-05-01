import util


x, y = util.extract_xy("experiment_out/soccer-joint-transformer/progress.csv", 1000)
util.plot(x, y)


