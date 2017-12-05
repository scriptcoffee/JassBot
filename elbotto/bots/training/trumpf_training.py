import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2
from keras.optimizers import SGD
from elbotto.bots.training.training import Training
from elbotto.bots.training.trumpf_converter import TrumpfCard, trumpf_converter

INPUT_LAYER = 37
FIRST_LAYER = 37
OUTPUT_LAYER = 7


class TrumpfTraining(Training):
    def __init__(self, name, log_path):
        super().__init__(name)

        self.tb_callback = keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=5, batch_size=32,
                                                       write_graph=False, write_grads=True, write_images=False,
                                                       embeddings_freq=0, embeddings_layer_names=None,
                                                       embeddings_metadata=None)
        self.save_model_and_weights("trumpf", "init")

    def define_model(self):
        self.q_model = Sequential()
        self.q_model.add(Dense(FIRST_LAYER, input_shape=(INPUT_LAYER,), kernel_initializer='uniform'))
        self.q_model.add(keras.layers.normalization.BatchNormalization())
        self.q_model.add(Activation("relu"))
        self.q_model.add(Dense(OUTPUT_LAYER, kernel_regularizer=l2(0.01)))
        self.q_model.add(Activation("softmax"))
        sgd = SGD(lr=0.005)
        self.q_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_squared_error', 'acc'])

    def train_the_model(self, start_handcards, trumpf):
        x = np.zeros((np.array(start_handcards).shape[0], INPUT_LAYER))
        y = np.zeros((np.array(trumpf).shape[0], OUTPUT_LAYER))
        input_list = []
        target_list = []
        for i in range(len(start_handcards)):
            input_list.append(create_input(start_handcards[i], trumpf[i - 1]))
            target_list.append(create_target(trumpf[i]))
        x[:, :] = input_list
        y[:, :] = target_list
        print("Input: {}".format(x))
        print("Output: {}".format(y))
        self.q_model.fit(x, y, validation_split=0.3, verbose=1, epochs=5, callbacks=[self.tb_callback])
        print("Learning Trumpf!")


def create_input(start_handcards, trumpf):
    if len(start_handcards) != 9:
        return None
    inputs = np.zeros((INPUT_LAYER,))
    for card in start_handcards:
        if inputs[card.id] == 1:
            return None
        inputs[card.id] = 1
    if not isinstance(trumpf, TrumpfCard):
        inputs[INPUT_LAYER - 1] = 0
    elif trumpf.mode == "SCHIEBE":
        inputs[INPUT_LAYER - 1] = 1
    return np.reshape(inputs, (1, INPUT_LAYER))


def choose_color(output_layer, trumpf):
    output_layer[trumpf.trumpf_color.value] = 1


def choose_obeabe(output_layer, trumpf):
    output_layer[4] = 1


def choose_undeufe(output_layer, trumpf):
    output_layer[5] = 1


def choose_schiebe(output_layer, trumpf):
    output_layer[6] = 1


CHOOSE_DICT = {"TRUMPF": choose_color,
               "OBEABE": choose_obeabe,
               "UNDEUFE": choose_undeufe,
               "SCHIEBE": choose_schiebe}


def create_target(trumpf):
    if not isinstance(trumpf, TrumpfCard):
        trumpf = trumpf_converter(trumpf)
    if trumpf is None:
        return trumpf
    target_layer = np.zeros((OUTPUT_LAYER,))
    CHOOSE_DICT[trumpf.mode](target_layer, trumpf)
    return np.reshape(target_layer, (1, OUTPUT_LAYER))
