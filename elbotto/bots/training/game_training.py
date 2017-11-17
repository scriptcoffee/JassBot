import keras
import numpy as np
import tensorflow as tf
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2
from keras.optimizers import SGD
from elbotto.bots.training.training import Training


class GameTraining(Training):
    def __init__(self, name):
        super().__init__(name)
        self.game_counter = 0

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        sess = tf.Session(config=config)
        k.set_session(sess)

        self.tb_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=5, batch_size=64,
                                                       write_graph=False, write_grads=True, write_images=False,
                                                       embeddings_freq=0, embeddings_layer_names=None,
                                                       embeddings_metadata=None)

    def define_model(self):
        self.q_model = Sequential()
        self.q_model.add(Dense(50, input_shape=(150,), kernel_initializer='uniform'))
        self.q_model.add(keras.layers.normalization.BatchNormalization())
        self.q_model.add(Activation("relu"))
        self.q_model.add(Dense(36, kernel_regularizer=l2(0.01)))
        sgd = SGD(lr=0.005)
        self.q_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_squared_error'])

    def train_the_model(self, hand_list, table_list, trumpf_list, target_list):
        x = np.zeros((np.array(hand_list).shape[0], 150))
        y = np.zeros((np.array(target_list).shape[0], 36))
        input_list = []
        target_layer = []
        for i in range(len(hand_list)):
            input_list.append(create_input(hand_list[i], table_list[i], trumpf_list[i]))
            target_layer.append(create_target(target_list[i]))

        x[:, :] = input_list
        y[:, :] = target_layer
        print("Input-Layer: " + str(x))
        print("Output-Layer: " + str(y))
        if len(y) > 1:
            if self.game_counter % 1000 == 0:
                self.q_model.fit(x, y, validation_split=0.1, verbose=1, callbacks=[self.tb_callback])
            else:
                self.q_model.fit(x, y, validation_split=0.1, verbose=1)
        self.game_counter += 1
        print("One Training-Part are finished!")


def create_input(hand_cards, table_cards, game_type):
    # 150 Inputs (4 x 36 possible cards per hand and 6 trumpf variations).
    # Partitional: 1-36 -> hand, 37-72 - first card on table, 73-108 - second card on table, 109-144 - third card on table, 145-150 set trumpf
    # Status: 0 - no info, 1 - know place of the card
    inputs = np.zeros((150,))
    for card in hand_cards:
        inputs[card.id] = 1
    for x in range(0, len(table_cards)):
        c = table_cards[x]
        inputs[c.id + (x + 1)*36] = 1
    if game_type.mode == "TRUMPF":
        inputs[game_type.trumpf_color.value + 4 * 36] = 1
    elif game_type.mode == "OBEABE":
        inputs[148] = 1
    elif game_type.mode == "UNDEUFE":
        inputs[149] = 1
    return np.reshape(inputs, (1, 150))


def create_target(target_card):
    comparison_list = np.zeros((36,))
    comparison_list[target_card.id] = 1
    return np.reshape(comparison_list, (1, 36))
