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

'''
|                               played cards
|    hand cards   |      table       |    player 0     |    player 1      |  player 2      |  player 3      |trumpf|
|-----------------|------------------|-----------------|------------------|----------------|----------------|------|

6 * 36 + 6 = 222

Split the game network in three learning parts - Three networks.
1. network for the first three cards (1.-3. cards) -> first_network
2. network for the second three cards (4.-6. cards) -> second_network
3. network for the last three cards (7.-9. cards) -> third_network
'''

INPUT_LAYER = 222
FIRST_LAYER = 670
SECOND_LAYER = 2010
THIRD_LAYER = 220
OUTPUT_LAYER = 36

CARD_SET = 36

pos_hand_cards = 0
pos_table = 1 * CARD_SET
pos_players_played_cards = [2 * CARD_SET, 3 * CARD_SET, 4 * CARD_SET, 5 * CARD_SET]
pos_trumpf = INPUT_LAYER - 6  # Alternativ calculation: 6 * CARD_SET

model_path = "./config/"


class GameTraining(Training):
    def __init__(self, main_network_name, log_path):
        self.network_list = ['first_network', 'second_network', 'third_network']
        for name in self.network_list:
            super().__init__('{}_{}'.format(main_network_name, name))
            self.game_counter = 0

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            k.set_session(sess)

            self.tb_callback = keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=5, batch_size=64,
                                                           write_graph=False, write_grads=True, write_images=False,
                                                           embeddings_freq=0, embeddings_layer_names=None,
                                                           embeddings_metadata=None)

            filename = 'game_{}'.format(name)
            self.save_model_and_weights(filename, "init")
            self.save_model("{}{}_model.h5".format(model_path, filename), network_name=name)

    def define_model(self):
        self.q_model = Sequential()
        self.q_model.add(Dense(FIRST_LAYER, activation='relu', input_shape=(INPUT_LAYER,),
                               kernel_initializer='uniform'))
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
        first_input_list = []
        first_target_layer = []
        second_input_list = []
        second_target_layer = []
        third_input_list = []
        third_target_layer = []

        for i in range(len(hand_list)):
            # Split into three list (first, second, third)
            amount_hand_cards = len(hand_list[i])
            if amount_hand_cards > 6:
                first_input_list.append(create_input(hand_list[i], table_list[i], played_card_list[i], trumpf_list[i]))
                first_target_layer.append(create_target(target_list[i]))
            elif amount_hand_cards < 4:
                third_input_list.append(create_input(hand_list[i], table_list[i], played_card_list[i], trumpf_list[i]))
                third_target_layer.append(create_target(target_list[i]))
            else:
                second_input_list.append(create_input(hand_list[i], table_list[i], played_card_list[i], trumpf_list[i]))
                second_target_layer.append(create_target(target_list[i]))

        first_x = np.zeros((np.array(first_input_list).shape[0], INPUT_LAYER))
        first_y = np.zeros((np.array(first_target_layer).shape[0], OUTPUT_LAYER))
        second_x = np.zeros((np.array(second_input_list).shape[0], INPUT_LAYER))
        second_y = np.zeros((np.array(second_target_layer).shape[0], OUTPUT_LAYER))
        third_x = np.zeros((np.array(third_input_list).shape[0], INPUT_LAYER))
        third_y = np.zeros((np.array(third_target_layer).shape[0], OUTPUT_LAYER))

        first_x[:, :] = first_input_list
        first_y[:, :] = first_target_layer
        second_x[:, :] = second_input_list
        second_y[:, :] = second_target_layer
        third_x[:, :] = third_input_list
        third_y[:, :] = third_target_layer

        np_xy_lists = [(first_x, first_y, self.network_list[0]), (second_x, second_y, self.network_list[1]),
                       (third_x, third_y, self.network_list[2])]

        for t in np_xy_lists:
            filepath = "{}game_{}_model.h5".format(model_path, t[2])

            self.load_model(filepath, t[2])
            if len(t[1]) > 1:
                if self.game_counter % 500 == 0:
                    self.q_model.fit(t[0], t[1], validation_split=0.1, epochs=10, verbose=1,
                                     callbacks=[self.tb_callback])
                else:
                    self.q_model.fit(t[0], t[1], validation_split=0.1, epochs=10, verbose=1)
            self.game_counter += 1
            self.save_model(filepath, network_name=t[2])

        print("One Training-Part are finished!")


def create_input(hand_cards, table_cards, played_cards, game_type):
    inputs = np.zeros((INPUT_LAYER,))

    if len(hand_cards) == 0:
        return None
    for card in hand_cards:
        if card is None:
            return None
        inputs[pos_hand_cards + card.id] = 1

    for x in range(len(table_cards)):
        card = table_cards[x]
        inputs[pos_table + card.id] = 1

    if played_cards is None:
        return None
    if len(played_cards) > 0 and isinstance(played_cards[0], Card):
        print('The played card list has a wrong format!')
        return None
    else:
        for player_of_cards in range(len(played_cards)):
            if played_cards[player_of_cards] is None:
                break
            for played_card in played_cards[player_of_cards]:
                inputs[pos_players_played_cards[player_of_cards] + played_card.id] = 1

    if game_type.mode == "TRUMPF":
        inputs[pos_trumpf + game_type.trumpf_color.value] = 1
    elif game_type.mode == "OBEABE":
        inputs[pos_trumpf + 4] = 1
    elif game_type.mode == "UNDEUFE":
        inputs[pos_trumpf + 5] = 1

    return np.reshape(inputs, (1, INPUT_LAYER))


def create_target(target_card):
    if not isinstance(target_card, Card):
        return None
    target_list = np.zeros((OUTPUT_LAYER,))
    target_list[target_card.id] = 1
    return np.reshape(target_list, (1, OUTPUT_LAYER))
