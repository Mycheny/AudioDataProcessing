import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import GRU
from keras import regularizers
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from keras.constraints import Constraint
from keras import backend as K


def my_crossentropy(y_true, y_pred):
    return K.mean(2 * K.abs(y_true - 0.5) * K.binary_crossentropy(y_pred, y_true), axis=-1)


def mymask(y_true):
    return K.minimum(y_true + 1., 1.)


def msse(y_true, y_pred):
    return K.mean(mymask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)


def mycost(y_true, y_pred):
    return K.mean(mymask(y_true) * (10 * K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(
        K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01 * K.binary_crossentropy(y_pred, y_true)), axis=-1)


def my_accuracy(y_true, y_pred):
    return K.mean(2 * K.abs(y_true - 0.5) * K.equal(y_true, K.round(y_pred)), axis=-1)


class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''

    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}


def get_model():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.42
    set_session(tf.Session(config=config))
    reg = 0.000001
    constraint = WeightClip(0.499)

    print('Build model...')
    main_input = Input(shape=(None, 34), name='main_input')
    tmp = Dense(24, activation='tanh', name='input_dense', kernel_constraint=constraint, bias_constraint=constraint)(
        main_input)
    vad_gru = GRU(24, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='vad_gru',
                  kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg),
                  kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(tmp)
    vad_output = Dense(1, activation='sigmoid', name='vad_output', kernel_constraint=constraint,
                       bias_constraint=constraint)(vad_gru)
    noise_input = keras.layers.concatenate([tmp, vad_gru, main_input], name="noise_input")
    noise_gru = GRU(48, activation='relu', recurrent_activation='sigmoid', return_sequences=True, name='noise_gru',
                    kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg),
                    kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(
        noise_input)
    denoise_input = keras.layers.concatenate([vad_gru, noise_gru, main_input], name="denoise_input")

    denoise_gru = GRU(96, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='denoise_gru',
                      kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg),
                      kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(
        denoise_input)

    denoise_output = Dense(22, activation='sigmoid', name='denoise_output', kernel_constraint=constraint,
                           bias_constraint=constraint)(denoise_gru)

    model = Model(inputs=main_input, outputs=[denoise_output, vad_output])
    # model.summary()

    model.compile(loss=[mycost, my_crossentropy],
                  metrics=[msse],
                  optimizer='adam', loss_weights=[10, 0.5])
    return model
