import keras
import numpy as np
from json import dump
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2
from keras.optimizers import SGD


class Training(object):

    def __init__(self, name):
        self.name = name
        self.t_model = self.define_model()
        self.tb_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=5, batch_size=12,
                                                       write_graph=False, write_grads=True, write_images=False,
                                                       embeddings_freq=0, embeddings_layer_names=None,
                                                       embeddings_metadata=None)

    @staticmethod
    def define_model():
        t_model = Sequential()
        t_model.add(Dense(36, input_shape=(36,), kernel_initializer='uniform'))
        t_model.add(keras.layers.normalization.BatchNormalization())
        t_model.add(Activation("relu"))
        t_model.add(Dense(6, kernel_regularizer=l2(0.01)))
        sgd = SGD(lr=0.005)
        t_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_squared_error'])
        return t_model

    def save_weights(self, path):
        self.t_model.save_weights(path)
        return print("The weights of your model saved.")

    def save_model(self, path, json=False):
        if json:
            model_json = self.t_model.to_json()
            with open(path, 'w') as f:
                dump(model_json, f)
            save_type = 'json'
        else:
            self.t_model.save(path)
            save_type = 'h5'
        return print("The model saved as " + save_type + ".")

    @staticmethod
    def create_input(start_handcards):
        inputs = np.zeros((36,))
        for card in start_handcards:
            inputs[card.id] = 1
        return np.reshape(inputs, (1, 36))

    @staticmethod
    def create_target(trumpf):
        # one item from input convert to a 6 output matrix about all trumpf modes
        output_layer = np.zeros((6,))
        print(str(trumpf.mode))
        if trumpf.mode == "TRUMPF":
            output_layer[trumpf.trumpf_color.value] = 1
            print(str(trumpf.trumpf_color.value))
        elif trumpf.mode == "OBEABE":
            output_layer[4] = 1
        elif trumpf.mode == "UNDEUFE":
            output_layer[5] = 1
        return np.reshape(output_layer, (1, 6))

    def train_the_model(self, start_handcards, trumpf):
        x = np.zeros((np.array(start_handcards).shape[0], 36))
        y = np.zeros((np.array(trumpf).shape[0], 6))
        input_list = []
        output_list = []
        print("Start_Handcards: " + str(start_handcards))
        for i in range(len(start_handcards)):
            input_list.append(self.create_input(start_handcards[i]))
            output_list.append(self.create_target(trumpf[i]))
        print(input_list)
        print(output_list)
        x[:, :] = input_list
        y[:, :] = output_list
        print("Input: " + str(x))
        print("Output: " + str(y))
        self.t_model.fit(x, y, validation_split=0.3, callbacks=[self.tb_callback])
        print("Learning Trumpf!")
