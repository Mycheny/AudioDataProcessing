#!/usr/bin/python

from __future__ import print_function
import h5py
import numpy as np
import os

from rnnoise.gru import get_model

model = get_model()
model.load_weights("./resources/model/model.hdf5")
train_data = ["./test/data/cafe", "./test/data/car", "./test/data/white"]*20
for path in train_data:
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        print(file_path)
        with h5py.File(file_path, 'r') as hf:
            all_data = hf['data'][:]

        window_size = 200

        nb_sequences = len(all_data) // window_size
        print(nb_sequences, ' sequences')
        x_train = all_data[:nb_sequences * window_size, 1:35]
        x_train = (np.reshape(x_train, (nb_sequences, window_size, 34))+100)

        y_train = np.copy(all_data[:nb_sequences * window_size, 35:57])
        y_train = np.reshape(y_train, (nb_sequences, window_size, 22))

        # noise_train = np.copy(all_data[:nb_sequences * window_size, 64:86])
        # noise_train = np.reshape(noise_train, (nb_sequences, window_size, 22))

        vad_train = np.copy(all_data[:nb_sequences * window_size, 0:1])  # Voice Activity Detection 语音激活检测
        vad_train = np.reshape(vad_train, (nb_sequences, window_size, 1))

        all_data = 0
        # x_train = x_train.astype('float32')
        # y_train = y_train.astype('float32')

        print(len(x_train), 'train sequences. x shape =', x_train.shape, 'y shape = ', y_train.shape)

        print('Train...')
        model.fit(x_train, [y_train, vad_train],
                  batch_size=32,
                  epochs=2,
                  validation_split=0.1)
        model.save("./resources/model/model.hdf5")

y = model.predict(x_train)

h5f = h5py.File("y0.h5", 'w')
h5f.create_dataset('d', data=y[0])
h5f = h5py.File("y1.h5", 'w')
h5f.create_dataset('d', data=y[1])
h5f.close()
