'''
    Normalize the data, save as ./data/normalized_data.npy
'''

import numpy as np
import os

instances = np.load('data/instances.npy')
normalized = np.zeros(shape=instances.shape)

mean = np.zeros((instances.shape[0],6))
std_dev = np.zeros((instances.shape[0],6))

for i in range(0,6):
    for j in range(0, instances.shape[0]):
        mean[j][i] = np.mean(instances[j,:,i])
        std_dev[j][i] = np.std(instances[j,:,i])

for i in range(0,6):
    for j in range(0, instances.shape[0]):
        normalized[j,:,i] = (instances[j,:,i] - mean[j][i])/std_dev[j][i]

np.save(os.path.join('data', 'normalized_data.npy'), normalized)