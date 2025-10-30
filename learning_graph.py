#%%
import numpy as np
from matplotlib import pyplot as plt

rewards = np.load('q-prob_rewards.npy')

plt.xlabel("episodes")
plt.ylabel("rewards")
plt.plot(rewards[:2000])
# %%
