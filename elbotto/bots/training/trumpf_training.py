import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2
from keras.optimizers import SGD
from elbotto.bots.training.training import Training


class TrumpfTraining(Training):
    def __init__(self, name):
        super().__init__(name)

        self.tb_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=5, batch_size=12,
                                                       write_graph=False, write_grads=True, write_images=False,
                                                       embeddings_freq=0, embeddings_layer_names=None,
                                                       embeddings_metadata=None)

    def define_model(self):
        self.q_model = Sequential()
        self.q_model.add(Dense(36, input_shape=(36,), kernel_initializer='uniform'))
        self.q_model.add(keras.layers.normalization.BatchNormalization())
        self.q_model.add(Activation("relu"))
        self.q_model.add(Dense(6, kernel_regularizer=l2(0.01)))
        self.q_model.add(Activation("softmax"))
        sgd = SGD(lr=0.005)
        self.q_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_squared_error', 'acc'])

    def train_the_model(self, start_handcards, trumpf):
        x = np.zeros((np.array(start_handcards).shape[0], 36))
        y = np.zeros((np.array(trumpf).shape[0], 6))
        input_list = []
        target_list = []
        print("Start_Handcards: " + str(start_handcards))
        for i in range(len(start_handcards)):
            input_list.append(create_input(start_handcards[i]))
            target_list.append(create_target(trumpf[i]))
        print(input_list)
        print(target_list)
        x[:, :] = input_list
        y[:, :] = target_list
        print("Input: " + str(x))
        print("Output: " + str(y))
        self.q_model.fit(x, y, validation_split=0.3, callbacks=[self.tb_callback])
        print("Learning Trumpf!")


def create_input(start_handcards):
    inputs = np.zeros((36,))
    for card in start_handcards:
        inputs[card.id] = 1
    return np.reshape(inputs, (1, 36))


def create_target(trumpf):
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
