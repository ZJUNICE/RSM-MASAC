import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


training_score = np.load('baseline_training_score.npy')
training_score = np.array(training_score).mean(axis=0)
training_score = training_score.reshape((5, 5))
sns.set()
ax = plt.figure()
ax = sns.heatmap(training_score, cmap=plt.cm.Blues)
plt.show()