'''
    Visualize some basic statistics of our dataset.
'''
import matplotlib.pyplot as plt
import numpy as np

def find_avg(gestures):
    '''
    Find the average sensor values over both time and gesture instances. Gestures = m x 100 x 6 np array.
    '''

    avg_time = np.zeros((gestures.shape[0],6))
    std_time = np.zeros((gestures.shape[0],6))
    avg_instance = np.zeros(6)
    std_instance = np.zeros(6)
    gest_num = ['ax','ay','az','wx','wy','wz']

    for i in range(0,6):
        for j in range(0,gestures.shape[0]):
            avg_time[j][i] = np.mean(gestures[j,:,i])
            std_time[j][i] = np.std(gestures[j,:,i])
        avg_instance[i] = np.mean(avg_time[:,i])
        std_instance[i] = np.std(std_time[:,i])

    # Plot
    plt.figure()
    plt.title("Average Features")
    plt.bar(gest_num, avg_instance, yerr=std_instance, capsize=5)
    plt.xlabel("Gesture")
    plt.ylabel("Average Feature")

    return

instances = np.load('data/instances.npy')
labels = np.load('data/labels.npy')

a_ind = np.squeeze(np.argwhere(labels == 'a'))
k_ind = np.squeeze(np.argwhere(labels == 'k'))
z_ind = np.squeeze(np.argwhere(labels == 'z'))

a = instances[a_ind,:,:]
k = instances[k_ind,:,:]
z = instances[z_ind,:,:]

find_avg(a)
find_avg(k)
find_avg(z)


