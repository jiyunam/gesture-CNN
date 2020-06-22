'''
    Save the data in the .csv file, save as a .npy file in ./data
'''

import numpy as np
import os

instances = []
labels = []
time = []

for dirName, subdirList, fileList in os.walk('data/unnamed_train_data'):
    for student in subdirList:
        for stu_dirName, stu_subdirList, stu_fileList in os.walk('data/unnamed_train_data/%s' % (student)):
            for file in stu_fileList:
                data = np.loadtxt('data/unnamed_train_data/%s/%s' % (student,file), delimiter=',')
                instances.append(data[:,1:])

                label, number = file.split("_")
                labels.append(label)

time = data[:,0]
np.save(os.path.join('data', 'instances.npy'), instances)
np.save(os.path.join('data', 'labels.npy'), labels)
np.save(os.path.join('data', 'time.npy'), time)
