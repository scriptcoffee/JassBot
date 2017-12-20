import keras
import numpy as np
import tensorflow as tf
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from elbotto.bots.training.training import Training
from elbotto.bots.training.card_parser import Card

INPUT_LAYER = 186
FIRST_LAYER = 560
SECOND_LAYER = 1680
THIRD_LAYER = 180
OUTPUT_LAYER = 36

CARD_SET = 36


class GameTraining(Training):
    def __init__(self, name, log_path):
        super().__init__(name)
        self.game_counter = 0

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        k.set_session(sess)

        self.tb_callback = keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=5, batch_size=64,
                                                       write_graph=False, write_grads=True, write_images=False,
                                                       embeddings_freq=0, embeddings_layer_names=None,
                                                       embeddings_metadata=None)
        self.save_model_and_weights("game", "init")

    def define_model(self):
        self.q_model = Sequential()
        self.q_model.add(Dense(FIRST_LAYER, activation='relu', input_shape=(INPUT_LAYER,), kernel_initializer='uniform'))
        self.q_model.add(BatchNormalization())
        self.q_model.add(Dense(SECOND_LAYER, activation='relu', kernel_initializer='uniform'))
        self.q_model.add(BatchNormalization())
        self.q_model.add(Dense(THIRD_LAYER, activation='relu', kernel_initializer='uniform'))
        self.q_model.add(BatchNormalization())
        self.q_model.add(Dense(OUTPUT_LAYER, activation='softmax', kernel_regularizer=l2(0.01)))
        adam = Adam(lr=0.005)
        self.q_model.compile(loss='categorical_crossentropy', optimizer=adam,
                             metrics=['categorical_accuracy', 'categorical_crossentropy', 'mean_squared_error', 'acc'])

    def train_the_model(self, hand_list, table_list, played_card_list, trumpf_list, target_list):
        x = np.zeros((np.array(hand_list).shape[0], INPUT_LAYER))
        y = np.zeros((np.array(target_list).shape[0], OUTPUT_LAYER))
        input_list = []
        target_layer = []
        for i in range(len(hand_list)):
            input_list.append(create_input(hand_list[i], table_list[i], played_card_list[i], trumpf_list[i]))
            target_layer.append(create_target(target_list[i]))

        x[:, :] = input_list
        y[:, :] = target_layer
        print("Input-Layer: {}".format(x))
        print("Output-Layer: {}".format(y))
        if len(y) > 1:
            if self.game_counter % 500 == 0:
                self.q_model.fit(x, y, validation_split=0.1, epochs=10, verbose=1, callbacks=[self.tb_callback])
            else:
                self.q_model.fit(x, y, validation_split=0.1, epochs=10, verbose=1)
        self.game_counter += 1
        print("One Training-Part are finished!")


def create_input(hand_cards, table_cards, played_cards, game_type):
    trumpf_offset = INPUT_LAYER - 6
    inputs = np.zeros((INPUT_LAYER,))

    if len(hand_cards) == 0:
        return None
    for card in hand_cards:
        if card is None:
            return None
        inputs[card.id] = 1

    for x in range(len(table_cards)):
        card = table_cards[x]
        inputs[card.id + ((x + 1) * CARD_SET)] = 1

    if len(played_cards) > 0 and isinstance(played_cards[0], Card):
        for played_card in played_cards:
            inputs[(4 * CARD_SET) + played_card.id] = 1
    else:
        for player_of_cards in range(len(played_cards)):
            if played_cards[player_of_cards] is None:
                break
            for played_card in played_cards[player_of_cards]:
                inputs[(4 * CARD_SET) + played_card.id] = 1

    if game_type.mode == "TRUMPF":
        inputs[game_type.trumpf_color.value + trumpf_offset] = 1
    elif game_type.mode == "OBEABE":
        inputs[trumpf_offset + 4] = 1
    elif game_type.mode == "UNDEUFE":
        inputs[trumpf_offset + 5] = 1
    return np.reshape(inputs, (1, INPUT_LAYER))


def create_target(target_card):
    if not isinstance(target_card, Card):
        return None
    target_list = np.zeros((OUTPUT_LAYER,))
    target_list[target_card.id] = 1
    return np.reshape(target_list, (1, OUTPUT_LAYER))
