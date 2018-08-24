import os
import keras
import numpy as np
import tensorflow as tf
from keras import backend as k
from keras.layers import Input, Dense
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

|   first_network   |   second_network  |   third_network   |
|--------...--------|--------...--------|--------...--------|

3 * 222 = 666

Every network has the same input and the same output layer, but the three game networks are independent of each other.

The output layer of each network has 36 neurons or 36 possible cards.
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

MODEL_INPUT_LAYER = 3 * INPUT_LAYER
MODEL_OUTPUT_LAYER = 3 * OUTPUT_LAYER
pos_input_networks = [0 * INPUT_LAYER, 1 * INPUT_LAYER, 2 * INPUT_LAYER]
pos_output_networks = [0 * OUTPUT_LAYER, 1 * OUTPUT_LAYER, 2 * OUTPUT_LAYER]

dir_path = os.path.dirname(os.path.abspath(__file__))
model_path = dir_path.join('config')


class GameTraining(Training):
    def __init__(self, main_network_name, log_path):
        self.network_names = ['first_network', 'second_network', 'third_network']
        self.dict_networks = {}
        super().__init__(main_network_name)
        self.game_counter = 0

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        k.set_session(sess)

        self.tb_callback = keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=5, batch_size=64,
                                                       write_graph=False, write_grads=True, write_images=False,
                                                       embeddings_freq=0, embeddings_layer_names=None,
                                                       embeddings_metadata=None)

        self.save_model_and_weights(main_network_name, "init")

    def define_model(self):
        for i, network_name in enumerate(self.network_names):
            print('out from network_list: ', network_name)
            start_layer = Input(shape=(INPUT_LAYER,))
            self.dict_networks['{}_input_layer'.format(network_name)] = start_layer
            first_layer = Dense(FIRST_LAYER, activation='relu', kernel_initializer='uniform')(start_layer)
            self.dict_networks['{}_first_layer'.format(network_name)] = first_layer
            second_layer = Dense(SECOND_LAYER, activation='relu', kernel_initializer='uniform')(first_layer)
            self.dict_networks['{}_second_layer'.format(network_name)] = second_layer
            third_layer = Dense(THIRD_LAYER, activation='relu', kernel_initializer='uniform')(second_layer)
            self.dict_networks['{}_third_layer'.format(network_name)] = third_layer
            last_layer = Dense(OUTPUT_LAYER, activation='softmax', kernel_regularizer=l2(0.01))(third_layer)
            self.dict_networks['{}_output_layer'.format(network_name)] = last_layer

        self.q_model = keras.models.Model(
            inputs=[self.dict_networks.get('first_network_input_layer'),
                    self.dict_networks.get('second_network_input_layer'),
                    self.dict_networks.get('third_network_input_layer')],
            outputs=[self.dict_networks.get('first_network_output_layer'),
                     self.dict_networks.get('second_network_output_layer'),
                     self.dict_networks.get('third_network_output_layer')])

        adam = Adam(lr=0.005)
        self.q_model.compile(loss='categorical_crossentropy', optimizer=adam,
                             metrics=['categorical_accuracy', 'categorical_crossentropy', 'mean_squared_error', 'acc'])

        print('model', self.q_model.summary())

    def train_the_model(self, hand_list, table_list, played_card_list, trumpf_list, target_list):

        input_list = []
        target_layer = []

        for i in range(len(hand_list)):
            # Split the list for the three trainable networks (first, second, third)
            pos_input_network, pos_output_network = choose_network(len(hand_list[i]))
            input_list.append(
                create_input(hand_list[i], table_list[i], played_card_list[i], trumpf_list[i], pos_input_network))
            target_layer.append(create_target(target_list[i], pos_output_network))

        x = np.zeros((np.array(input_list).shape[0], MODEL_INPUT_LAYER))
        y = np.zeros((np.array(target_layer).shape[0], MODEL_OUTPUT_LAYER))

        print('x:', x.shape)
        print('y:', y.shape)

        x[:, :] = input_list
        y[:, :] = target_layer

        x = x.reshape((3, len(x), 222))
        y = y.reshape((3, len(y), 36))

        if self.game_counter % 500 == 0:
            self.q_model.fit({'input_1': x[0], 'input_2': x[1], 'input_3': x[2]},
                             {'dense_4': y[0], 'dense_8': y[1], 'dense_12': y[2]},
                             validation_split=0.1, epochs=10, verbose=1, callbacks=[self.tb_callback])
        else:
            self.q_model.fit({'input_1': x[0], 'input_2': x[1], 'input_3': x[2]},
                             {'dense_4': y[0], 'dense_8': y[1], 'dense_12': y[2]},
                             validation_split=0.1, epochs=10, verbose=1)

        self.game_counter += 1

        print("One Training-Part are finished!")


def create_input(hand_cards, table_cards, played_cards, game_type, pos_network=None):
    inputs = np.zeros((MODEL_INPUT_LAYER,))

    amount_hand_cards = len(hand_cards)

    if amount_hand_cards == 0:
        return None

    if pos_network is None:
        pos_network, _ = choose_network(amount_hand_cards)
        if pos_network is None:
            return None

    for card in hand_cards:
        if card is None:
            return None
        inputs[pos_network + pos_hand_cards + card.id] = 1

    for x in range(len(table_cards)):
        card = table_cards[x]
        inputs[pos_network + pos_table + card.id] = 1

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
                inputs[pos_network + pos_players_played_cards[player_of_cards] + played_card.id] = 1

    if game_type.mode == "TRUMPF":
        inputs[pos_network + pos_trumpf + game_type.trumpf_color.value] = 1
    elif game_type.mode == "OBEABE":
        inputs[pos_network + pos_trumpf + 4] = 1
    elif game_type.mode == "UNDEUFE":
        inputs[pos_network + pos_trumpf + 5] = 1

    return np.reshape(inputs, (1, MODEL_INPUT_LAYER))


def create_target(target_card, pos_network):
    if not isinstance(target_card, Card):
        return None
    target_list = np.zeros((MODEL_OUTPUT_LAYER,))
    target_list[pos_network + target_card.id] = 1
    return np.reshape(target_list, (1, MODEL_OUTPUT_LAYER))


def choose_network(amount_hand_cards):
    if not isinstance(amount_hand_cards, int):
        return None, None
    if 10 > amount_hand_cards > 6:
        return pos_input_networks[0], pos_output_networks[0]
    elif 0 < amount_hand_cards < 4:
        return pos_input_networks[2], pos_output_networks[2]
    elif 3 < amount_hand_cards < 7:
        return pos_input_networks[1], pos_output_networks[1]
    else:
        return None, None
