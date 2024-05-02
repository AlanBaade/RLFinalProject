import matplotlib.pyplot as plt
import numpy as np

# TRAIN
categories = ['TRANSFORMER SMALL', 'LINEAR SMALL', 'JOINT A2C', 'INDIVIDUAL A2C', 'JOINT LARGE', 'INDIV LARGE', 'LINEAR LARGE', 'INDIVIDUAL SMALL', 'JOINT SMALL', 'INDIVIDUAL SMALL AGENT REWARD', 'JOINT SMALL AGENT REWARD', 'BASELINE SMALL', 'BASELINE LARGE']
categories2 = [i.replace(" ", "\n") for i in categories]
means = [-69.40859599470019, -76.97444327570696, -82.9, -83.22699541716968, -72.36494543736231, 79.88207760525599, -91.85969078019036, -40.7119575050724, -34.113095743898654, -32.22437981336122, -46.411184168725434, -90.8977253953315, -91.14976963302594]
stdvs = [1.1682297923822627, 0.6705170658942178, 0.12232129822725066, 0.18951234797972322, 3.620921208145924, 5.190058445227391, 0.8388676490962623, 1.4208599345056667, 0.9979853593684919, 0.7738606436736315, 1.2601318239399433, 0.24600416091020977, 0.7682228966715968]

# TEST
# categories = ['TRANSFORMER SMALL', 'LINEAR SMALL', 'JOINT A2C', 'INDIVIDUAL A2C', 'JOINT LARGE', 'INDIV LARGE', 'LINEAR LARGE', 'INDIVIDUAL SMALL', 'JOINT SMALL', 'INDIVIDUAL SMALL AGENT REWARD', 'JOINT SMALL AGENT REWARD', 'BASELINE SMALL', 'BASELINE LARGE']
# categories2 = [i.replace(" ", "\n") for i in categories]
# means = [-85.58972211619754, -80.67423028842711, -84.925, -85.57357595173146, -77.98629362748662, 49.784356770050266, -90.90717670274203, -49.97112247367445, -50.329459137173934, -54.951104889587384, -68.41699514063775, -92.75785808856077, -93.56361170927437]
# stdvs = [1.1288103535895966, 0.7242940066905408, 0.11155155758661554, 0.16974881255063182, 2.8036228722776957, 7.686183174782502, 0.8063941229198843, 1.9713920888935395, 2.2387493646079224, 2.5341107807812877, 2.3261220543635934, 0.23728063116640258, 0.7576901012202326]


def s(l):
  return np.array([l[i] for i in indices])




# SMALL model comparison
indices = [7, 8, 0, 1]
plt.figure(figsize=(5, 4))
plt.bar(s(categories2), s(means)+100, width=.5, bottom=-100)
plt.errorbar(s(categories2), s(means), yerr=s(stdvs), fmt='none', capsize=15, ecolor="black")
plt.ylim(bottom=-100)
plt.ylabel("Mean Reward")
plt.axhline(means[11], linestyle='--')
plt.tight_layout()
plt.savefig("small")

# SMALL reward ablation
indices = [7, 8, 9, 10]
plt.figure(figsize=(5, 4))
plt.bar(s(categories2), s(means)+100, width=.5, bottom=-100)
plt.errorbar(s(categories2), s(means), yerr=s(stdvs), fmt='none', capsize=15, ecolor="black")
plt.ylim(bottom=-100)
plt.ylabel("Mean Reward")
plt.axhline(means[11], linestyle='--')
plt.tight_layout()
plt.savefig("rew")

# LARGE model comparison
indices = [5, 4, 6]
plt.figure(figsize=(4, 4))
plt.bar(s(categories2), s(means)+100, width=.5, bottom=-100)
plt.errorbar(s(categories2), s(means), yerr=s(stdvs), fmt='none', capsize=15, ecolor="black")
plt.ylim(bottom=-100)
plt.ylabel("Mean Reward")
plt.axhline(means[12], linestyle='--')
plt.tight_layout()
plt.savefig("large")

# algo ablation
indices = [7, 8, 3, 2]
cats = ["INDIVIDUAL\nSMALL\nPPO","JOINT\nSMALL\nPPO","INDIVIDUAL\nSMALL\nA2C","JOINT\nSMALL\nA2C"]
plt.figure(figsize=(5, 4))
plt.bar(cats, s(means)+100, width=.5, bottom=-100)
plt.errorbar(cats, s(means), yerr=s(stdvs), fmt='none', capsize=15, ecolor="black")
plt.ylim(bottom=-100)
plt.ylabel("Mean Reward")
plt.axhline(means[11], linestyle='--')
plt.tight_layout()
plt.savefig("algo")

