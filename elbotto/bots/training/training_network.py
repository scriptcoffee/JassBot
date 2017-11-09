import keras
import numpy as np
import tensorflow as tf
from keras import backend as k
from json import dump
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2
from keras.optimizers import SGD


class Training(object):
    def __init__(self, name):
        self.name = name

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        sess = tf.Session(config=config)
        k.set_session(sess)

        self.q_model = self.define_model()
        self.tb_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=5, batch_size=9,
                                                       write_graph=False, write_grads=True, write_images=False,
                                                       embeddings_freq=0, embeddings_layer_names=None,
                                                       embeddings_metadata=None)

    @staticmethod
    def define_model():
        q_model = Sequential()
        q_model.add(Dense(38, input_shape=(42,), kernel_initializer='uniform'))
        q_model.add(keras.layers.normalization.BatchNormalization())
        q_model.add(Activation("relu"))
        q_model.add(Dense(36, kernel_regularizer=l2(0.01)))
        sgd = SGD(lr=0.005)
        q_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_squared_error'])
        return q_model

    def save_weights(self, path):
        self.q_model.save_weights(path)
        return print("The weights of your model saved.")

    def save_model(self, path, json=False):
        if json:
            model_json = self.q_model.to_json()
            with open(path, 'w') as f:
                dump(model_json, f)
            save_type = 'json'
        else:
            self.q_model.save(path)
            save_type = 'h5'
        return print("The model saved as " + save_type + ".")

    @staticmethod
    def create_input(hand_cards, table_cards, game_type):
        # 36 Inputs (one per card).
        # Status: 0 - no info, 1 - in hand, 2 - first card on table, 3 - second card on table, 4 - third card on table
        inputs = np.zeros((42,))
        for card in hand_cards:
            inputs[card.id] = 1
        for x in range(0, len(table_cards)):
            c = table_cards[x]
            inputs[c.id] = x + 2
        if game_type.mode == "TRUMPF":
            inputs[game_type.trumpf_color.value + 36] = 1
        elif game_type.mode == "OBEABE":
            inputs[40] = 1
        elif game_type.mode == "UNDEUFE":
            inputs[41] = 1
        return np.reshape(inputs, (1, 42))

    @staticmethod
    def create_target(target_card):
        # one item from input convert to a 36 output matrix for learning about differenz cards
        comparison_list = np.zeros((36,))
        comparison_list[target_card.id] = 1
        return np.reshape(comparison_list, (1, 36))

    def train_the_model(self, hand_list, table_list, trumpf_list, target_list):
        x = np.zeros((np.array(hand_list).shape[0], 42))
        y = np.zeros((np.array(target_list).shape[0], 36))
        input_list = []
        target_layer = []
        for i in range(len(hand_list)):
            input_list.append(self.create_input(hand_list[i], table_list[i], trumpf_list[i]))
            target_layer.append(self.create_target(target_list[i]))

        x[:, :] = input_list
        y[:, :] = target_layer
        print("Input-Layer: " + str(x))
        print("Output-Layer: " + str(y))
        self.q_model.fit(x, y, validation_split=0.1, verbose=0, callbacks=[self.tb_callback])
        print("One Training-Part are finished!")
