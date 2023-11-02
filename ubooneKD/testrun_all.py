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
from models import TeacherAutoencoder, CicadaV1, CicadaV2
import gc

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    teacher = TeacherAutoencoder((864, 64, 1)).get_model()
    teacher.compile(optimizer = keras.optimizers.Adam(learning_rate=0.001), loss = 'mse')

# Open the HDF5 file

ntimeticks = 6400
nwire = 3456
f_downsample = 10
h_split = 10
v_split = 4
nbatch = 32

adccutoff = 10.*f_downsample/10.
adcsaturation = 100.*f_downsample/10.

train_ratio = 0.5
val_ratio = 0.1
test_ratio = 1 - train_ratio - val_ratio

for i in range(18):  # 0 to 17
    file_path = f"/nevis/westside/data/sc5303/Data/bnb_WithWire_{i:02d}.h5"

    with h5py.File(file_path, 'r') as hf:
        for startEvt in range(0,len(hf['wire_table']['adc'])//13056-nbatch,nbatch):
            print(f'run:{startEvt//nbatch}')
            
            planeadcs = [hf['wire_table']['adc'][4800+Evt*8256:8256+Evt*8256] for Evt in range(startEvt, startEvt+nbatch)]

            for p in range(0,nbatch):
                planeadcs[p] = block_reduce(planeadcs[p], block_size=(1,f_downsample), func=np.sum)

            for p in range(0,nbatch):
                planeadcs[p][planeadcs[p]<adccutoff] = 0
                planeadcs[p][planeadcs[p]>adcsaturation] = adcsaturation

            X = np.array(np.split(np.array(planeadcs), h_split, axis=2))
            X = np.array(np.split(np.array(X), v_split, axis=2))
            X = np.reshape(X, (-1,nwire//v_split,ntimeticks//h_split//f_downsample,1))

            print('X shape: ' + str(X.shape))

            # Create an array of original indices
            original_indices = np.arange(X.shape[0])

            # Perform the train/test split
            X_train_val_indices, X_test_indices = train_test_split(original_indices, test_size=test_ratio)
            X_train_indices, X_val_indices = train_test_split(X_train_val_indices, test_size=val_ratio / (val_ratio + train_ratio))
            # Use the indices to access the corresponding data
            X_train = X[X_train_indices]
            X_val = X[X_val_indices]

            print('X_train shape: ' + str(X_train.shape))
            print('X_val   shape: ' + str(X_val.shape))

            history = teacher.fit(X_train, X_train,
                            epochs = int(h_split*v_split*train_ratio),
                            validation_data = (X_val, X_val),
                            batch_size = int(nbatch))
            
            del X_train_val_indices, X_val, X_train, X_train_indices, X_val_indices
            del planeadcs, X, history
            tf.keras.backend.clear_session()
            gc.collect() #garbage collector collect

teacher.save(f'/nevis/westside/data/sc5303/saved_models/teacher_tiles{h_split}X{v_split}_full17')
