import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import roc_curve, auc
import qkeras
from qkeras import *
from skimage.measure import block_reduce

# Open the HDF5 file
file_path = '/nevis/westside/data/sc5303/Data/bnb_WithWire_16.h5'

with h5py.File(file_path, 'r') as hf:
    # Access the "wire_table" group
    wire_table_group = hf['wire_table']
    startEvt = 0
    endEvt = 100
    nEvt = endEvt - startEvt
    # Define the start and end indices for the subset you want to load

    # Load the subsets of the data using slicing and convert them to DataFrames
    event_ids = [wire_table_group['event_id'][Evt*8256] for Evt in range(startEvt, endEvt)]
    planeadcs = [wire_table_group['adc'][4800+Evt*8256:8256+Evt*8256] for Evt in range(startEvt, endEvt)]

    ntimeticks=6400
    nplanes=len(planeadcs)


    f_downsample = 10
    for p in range(0,nplanes):
        planeadcs[p] = block_reduce(planeadcs[p], block_size=(1,f_downsample), func=np.sum)

    adccutoff = 10.*f_downsample/10.
    adcsaturation = 100.*f_downsample/10.
    for p in range(0,nplanes):
        planeadcs[p][planeadcs[p]<adccutoff] = 0
        planeadcs[p][planeadcs[p]>adcsaturation] = adcsaturation

f_split = 10
X = np.array(np.split(np.array(planeadcs), f_split, axis=2))
print(np.shape(planeadcs))
X = np.reshape(X, (-1,3456,640//f_split,1))
print('X      shape: ' + str(X.shape))

full_split = 1
full = np.array(np.split(np.array(planeadcs), full_split, axis=2))
full = np.reshape(full, (-1,3456,640//full_split,1))
print('full      shape: ' + str(full.shape))

train_ratio = 0.5
val_ratio = 0.1
test_ratio = 1 - train_ratio - val_ratio

# Create an array of original indices
original_indices = np.arange(X.shape[0])

# Perform the train/test split
X_train_val_indices, X_test_indices = train_test_split(original_indices, test_size=test_ratio, random_state=42)
X_train_indices, X_val_indices = train_test_split(X_train_val_indices, test_size=val_ratio / (val_ratio + train_ratio), random_state=42)
del X_train_val_indices
# Use the indices to access the corresponding data
X_test = X[X_test_indices]
X_train = X[X_train_indices]
X_val = X[X_val_indices]

# X_train_val, X_test = train_test_split(X, test_size = test_ratio, random_state = 42)
# X_train, X_val = train_test_split(X_train_val, test_size = val_ratio/(val_ratio + train_ratio), random_state = 42)
# del X_train_val

print('X_train shape: ' + str(X_train.shape))
print('X_val   shape: ' + str(X_val.shape))
print('X_test  shape: ' + str(X_test.shape))

encoder_input = tf.keras.Input(shape=(3456,64,1), name='input')

encoder = layers.Conv2D(20, (3,3), strides=1, padding='same', name='conv2d_1')(encoder_input)
encoder = layers.Activation('relu', name='relu_1')(encoder)
encoder = layers.AveragePooling2D((2,2), name='pool_1')(encoder)
encoder = layers.Conv2D(30, (3,3), strides=1, padding='same', name='conv2d_2')(encoder)
encoder = layers.Activation('relu', name='relu_2')(encoder)
encoder = layers.Flatten(name='flatten')(encoder)

encoder_output = layers.Dense(80, activation='relu', name='latent')(encoder)

encoder = tf.keras.models.Model(encoder_input, encoder_output)

decoder = layers.Dense(1728*32*30, name='dense')(encoder_output)
decoder = layers.Reshape((1728,32,30), name='reshape2')(decoder)
decoder = layers.Activation('relu', name='relu_3')(decoder)
decoder = layers.Conv2D(30, (3,3), strides=1, padding='same', name='conv2d_3')(decoder)
decoder = layers.Activation('relu', name='relu_4')(decoder)
decoder = layers.UpSampling2D((2,2), name='upsampling')(decoder)
decoder = layers.Conv2D(20, (3,3), strides=1, padding='same', name='conv2d_4')(decoder)
decoder = layers.Activation('relu', name='relu_5')(decoder)

decoder_output = layers.Conv2D(1, (3,3), activation='relu', strides=1, padding='same', name='output')(decoder)

teacher = tf.keras.Model(encoder_input, decoder_output)
teacher.compile(optimizer = keras.optimizers.Adam(learning_rate=0.001), loss = 'mse')

history = teacher.fit(X_train, X_train,
                      epochs = 40,
                      validation_data = (X_val, X_val),
                      batch_size = 32)

X_train_predict_teacher = teacher.predict(X_train)
X_val_predict_teacher = teacher.predict(X_val)
X_test_predict_teacher = teacher.predict(X_test)

def loss(y_true, y_pred, choice):
    if choice == 'mse':
        loss = np.mean((y_true - y_pred)**2, axis = (1,2,3))
        return loss
    
X_train_loss_teacher = loss(X_train, X_train_predict_teacher, 'mse')
X_val_loss_teacher = loss(X_val, X_val_predict_teacher, 'mse')
X_test_loss_teacher = loss(X_test, X_test_predict_teacher, 'mse')

X_predict_teacher = teacher.predict(X)
X_loss_teacher = loss(X, X_predict_teacher, 'mse')

# Use boolean indexing to get the indices of values over the threshold
indices = np.where(X_loss_teacher > 50)

# Convert the indices to a list (if needed)
indices_list = indices[0].tolist()

zmax = adcsaturation

# Turn off interactive plotting
plt.ioff()

for evt in indices_list:
    fig, ax1 = plt.subplots(1, 1, figsize=(20, 15), dpi=600)
    true_indice = evt
    
    # Transpose the image
    transposed_image = X[true_indice][:, :, 0].T

    im1 = ax1.imshow(transposed_image, vmin=0, vmax=zmax, origin='lower', cmap='jet')
    ax1.set_title(f"Plane 2 run{event_ids[true_indice%nEvt][0]}, subrun{event_ids[true_indice%nEvt][1]}, event{event_ids[true_indice%nEvt][2]}, line:{true_indice//nEvt}, anomaly score: {X_loss_teacher[true_indice]:.2f}")
    ax1.set_xlabel("Wire")
    ax1.set_ylabel("Time Tick")

    # Save the figure without displaying it
    fig.savefig(f'output/strips64/teacher/{event_ids[true_indice%nEvt]}.png')
    # Close the figure to release resources
    plt.close(fig)


    fig, ax1 = plt.subplots(1, 1, figsize=(20, 15), dpi=600)
    true_indice = evt
    
    # Transpose the image
    transposed_image = full[true_indice%nEvt][:, :, 0].T

    im1 = ax1.imshow(transposed_image, vmin=0, vmax=zmax, origin='lower', cmap='jet')
    ax1.set_title(f"Plane 2 run{event_ids[true_indice%nEvt][0]}, subrun{event_ids[true_indice%nEvt][1]}, event{event_ids[true_indice%nEvt][2]}, line:{true_indice//nEvt}, anomaly score: {X_loss_teacher[true_indice]:.2f}")
    ax1.set_xlabel("Wire")
    ax1.set_ylabel("Time Tick")
    ax1.axhline((true_indice//nEvt)*64, linewidth=0.5, color='yellow', ls='-')
    ax1.axhline((true_indice//nEvt+1)*64, linewidth=0.5, color='yellow', ls='-')

    # Save the figure without displaying it
    fig.savefig(f'output/strips64/full/{event_ids[true_indice%nEvt]}.png')
    # Close the figure to release resources
    plt.close(fig)

# Turn interactive plotting back on
plt.ion()

# v2
x_in = layers.Input(shape=(3456*ntimeticks//f_split//f_downsample,), name="In")
x = layers.Reshape((3456,ntimeticks//f_split//f_downsample,1), name='reshape')(x_in)

x = QConv2D(3,(3,3), strides=2, padding="valid", use_bias=False,
            kernel_quantizer=quantized_bits(16,4,1,alpha='auto'), name='conv')(x)
x = QActivation('quantized_relu(16,4)', name='relu1')(x)
x = layers.Flatten(name='flatten')(x)
x = QDense(20, kernel_quantizer=quantized_bits(16,4,1,alpha='auto'),
           use_bias=False, name='dense1')(x)
x = QActivation('quantized_relu(16,4)', name='relu2')(x)
x = QDense(1, kernel_quantizer=quantized_bits(16,2,1,alpha='auto'),
           use_bias=False, name='output')(x)

student = tf.keras.models.Model(x_in, x)
student.compile(optimizer = 'adam', loss = 'mse')

history = student.fit(X_train.reshape((-1,3456*ntimeticks//f_split//f_downsample,1)), X_train_loss_teacher,
                      epochs = 30,
                      validation_data = (X_val.reshape((-1,3456*ntimeticks//f_split//f_downsample,1)), X_val_loss_teacher),
                      batch_size = 128)