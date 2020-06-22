'''
    Visualize some samples.
'''
import matplotlib.pyplot as plt
import numpy as np

# choose to plot a and z
instances = np.load('data/instances.npy')

# taking 3 separate instances of each letter
gest_inst_a = [0,130,5486]              # using np.argwhere(labels == 'a')
gest_inst_z = [125,255,5585]
[data_a_1,data_a_2,data_a_3] = [instances[gest_inst_a[0]],instances[gest_inst_a[1]],instances[gest_inst_a[2]]]
[data_z_1,data_z_2,data_z_3] = [instances[gest_inst_z[0]],instances[gest_inst_z[1]],instances[gest_inst_z[2]]]

# store each sensor values for both instances
[a_sensor_val_1,a_sensor_val_2,a_sensor_val_3] = [[],[],[]]
[z_sensor_val_1,z_sensor_val_2,z_sensor_val_3] = [[],[],[]]
sensor_value_number = np.arange(1,101,1)

a_sensor_vals = [a_sensor_val_1,a_sensor_val_2,a_sensor_val_3]
z_sensor_vals = [z_sensor_val_1,z_sensor_val_2,z_sensor_val_3]

for i in range(0,6):
    a_sensor_val_1.append(data_a_1[:,i])
    a_sensor_val_2.append(data_a_2[:,i])
    a_sensor_val_3.append(data_a_3[:,i])
    z_sensor_val_1.append(data_z_1[:,i])
    z_sensor_val_2.append(data_z_2[:,i])
    z_sensor_val_3.append(data_z_3[:,i])

# Plot figures
legend_labels = ['$a_x$','$a_y$','$a_z$','$w_x$','$w_y$','$w_z$']

for i in range(0,3):
    plt.figure()
    for j in range(0,6):
        plt.plot(sensor_value_number, a_sensor_vals[i][j],label='%s'%(legend_labels[j]))
    plt.legend()
    plt.title('Sensor Values for Letter "a" for Gesture Instance %s' %(gest_inst_a[i]))
    plt.xlabel('Sensor Value Number')
    plt.ylabel('Acceleration (m/s^2)/Rate of Rotation (rad/sec)')

for i in range(0,3):
    plt.figure()
    for j in range(0,6):
        plt.plot(sensor_value_number, z_sensor_vals[i][j],label='%s'%(legend_labels[j]))
    plt.legend()
    plt.title('Sensor Values for Letter "z" for Gesture Instance %s' %(gest_inst_z[i]))
    plt.xlabel('Sensor Value Number')
    plt.ylabel('Acceleration (m/s^2)/Rate of Rotation (rad/sec)')