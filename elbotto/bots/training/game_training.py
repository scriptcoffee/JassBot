import keras
import numpy as np
import tensorflow as tf
from keras import backend as k
from keras.layers import Input, Dense, Add, Concatenate
from keras.regularizers import l2
from keras.optimizers import Adam
from elbotto.bots.training.training import Training
from elbotto.bots.training.card_parser import Card

'''
|  36 hand cards  |   table cards (36 cards per player)   |   history (36 cards per stich per player)    | trumpf |
|-----------------|---------------------------------------|----------------------------------------------|--------|


|          36 table cards per each player           |
|---------------------------------------------------|
|| player 1  |  player 2  |  player 3  |  player 4 ||
|------------|------------|------------|------------|
-->> This is the current stich, because that are the most important cards for the current desicion
-->> That is the reason too, why we make the history just to 8 stichs 


|          36 cards per stich per player (history of the stichs from the current game without the last stich)         | 
|---------------------------------------------------------------------------------------------------------------------|
|           stich 1         ||           stich 2         ||           stich 3         |...|           stich 8         |
|---------------------------||---------------------------||---------------------------|...|---------------------------|
|   p1 |   p2 |   p3 |   p4 ||   p1 |   p2 |   p3 |   p4 ||   p1 |   p2 |   p3 |   p4 |...|   p1 |   p2 |   p3 |   p4 |


36 + 4 * 36 + 8 * 4 * 36 + 6 = 1'338
'''

CARD_LAYER = 36
TRUMPF_LAYER = 6

INPUT_LAYER = 1338
FIRST_LAYER = 350
SECOND_LAYER = 1030
THIRD_LAYER = 3080
FOURTH_LAYER = 520
FIFTH_LAYER = 85
OUTPUT_LAYER = 36

CARD_SET = 36

# the last stich lie on the table, so is the last one not part of the history
amount_stich_history = 8
pos_hand_cards = 0
pos_player_on_table = [1 * CARD_SET, 2 * CARD_SET, 3 * CARD_SET, 4 * CARD_SET]

amount_players = 4
amount_card_sets = 37


def generate_player_per_stich(start_range, end_range):
    player_collector = []
    pos_players_in_stich = []

    for player in range(start_range, end_range):
        player_collector.append(player * CARD_SET)
        if len(player_collector) == amount_players or player == end_range:
            pos_players_in_stich.append(player_collector)
            player_collector = []

    return pos_players_in_stich


pos_players_per_stich = generate_player_per_stich(amount_players + 1, amount_card_sets)

pos_trumpf = INPUT_LAYER - 6  # Alternativ calculation: 6 * CARD_SET


class GameTraining(Training):
    def __init__(self, name, log_path):
        self.dict_networks = {}
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

        hand_card_input_layer = Input(shape=(CARD_LAYER,))
        self.dict_networks['hand_card_input_layer'] = hand_card_input_layer
        hand_card_output_layer = Dense(CARD_LAYER, activation='relu', kernel_initializer='uniform')(
            hand_card_input_layer)
        self.dict_networks['hand_card_output_layer'] = hand_card_output_layer

        for player_on_table in range(amount_players):
            table_input_layer = Input(shape=(CARD_LAYER,))
            self.dict_networks['{}.player_table_input_layer'.format(player_on_table)] = table_input_layer
            table_first_layer = Dense(CARD_LAYER, activation='relu', kernel_initializer='uniform')(table_input_layer)
            self.dict_networks['{}.player_table_first_layer'.format(player_on_table)] = table_first_layer

        table_second_layer = Add()(self.give_dict_as_list(['{}.player_table_first_layer'.format(x)
                                                           for x in range(amount_players)], self.dict_networks))
        self.dict_networks['table_second_layer'] = table_second_layer
        table_output_layer = Dense(CARD_LAYER)(table_second_layer)
        self.dict_networks['table_output_layer'] = table_output_layer

        for i in range(amount_stich_history):
            player_counter = 0
            while player_counter < amount_players:
                stich_input_layer = Input(shape=(CARD_LAYER,))
                self.dict_networks['{}.stich_{}_player_input_layer'.format(i, player_counter)] = stich_input_layer
                stich_first_layer = Dense(CARD_LAYER, activation='relu', kernel_initializer='uniform')(
                    stich_input_layer)
                self.dict_networks['{}.stich_{}_player_first_layer'.format(i, player_counter)] = stich_first_layer
                player_counter += 1

            stich_second_layer = Add()(self.give_dict_as_list(['{}.stich_{}_player_first_layer'.format(i, y)
                                                               for y in range(amount_players)], self.dict_networks))
            self.dict_networks['{}.stich_second_layer'.format(i)] = stich_second_layer
        stich_output_layer = Add()(self.give_dict_as_list(['{}.stich_second_layer'.format(x)
                                                           for x in range(amount_stich_history)], self.dict_networks))
        self.dict_networks['stich_output_layer'] = stich_output_layer
        history_output_layer = Dense(CARD_LAYER, activation='relu', kernel_initializer='uniform')(stich_output_layer)
        self.dict_networks['history_output_layer'] = history_output_layer

        trumpf_input_layer = Input(shape=(TRUMPF_LAYER,))
        self.dict_networks['trumpf_input_layer'] = trumpf_input_layer
        trumpf_output_layer = Dense(TRUMPF_LAYER, activation='relu', kernel_initializer='uniform')(trumpf_input_layer)
        self.dict_networks['trumpf_output_layer'] = trumpf_output_layer

        input_layer = Concatenate()([self.dict_networks.get('hand_card_output_layer'),
                                     self.dict_networks.get('table_output_layer'),
                                     self.dict_networks.get('history_output_layer'),
                                     self.dict_networks.get('trumpf_output_layer')])
        self.dict_networks['input_layer'] = input_layer
        first_layer = Dense(FIRST_LAYER, activation='relu', kernel_initializer='uniform')(input_layer)
        self.dict_networks['first_layer'] = first_layer
        second_layer = Dense(SECOND_LAYER, activation='relu', kernel_initializer='uniform')(first_layer)
        self.dict_networks['second_layer'] = second_layer
        third_layer = Dense(THIRD_LAYER, activation='relu', kernel_initializer='uniform')(second_layer)
        self.dict_networks['third_layer'] = third_layer
        fourth_layer = Dense(FOURTH_LAYER, activation='relu', kernel_initializer='uniform')(third_layer)
        self.dict_networks['fourth_layer'] = fourth_layer
        fifth_layer = Dense(FIFTH_LAYER, activation='relu', kernel_initializer='uniform')(fourth_layer)
        self.dict_networks['fifth_layer'] = fifth_layer
        output_layer = Dense(OUTPUT_LAYER, activation='softmax', kernel_regularizer=l2(0.01))(fifth_layer)
        self.dict_networks['output_layer'] = output_layer

        self.q_model = keras.models.Model(
            inputs=[self.dict_networks.get('hand_card_input_layer')] +
                                       self.give_dict_as_list(['{}.player_table_input_layer'.format(x)
                                                               for x in range(amount_players)], self.dict_networks) +
                                       self.give_dict_as_list(['{}.stich_{}_player_input_layer'.format(x, y)
                                                               for x in range(amount_stich_history)
                                                               for y in range(amount_players)], self.dict_networks) +
                                       [self.dict_networks.get('trumpf_input_layer')],
            outputs=[self.dict_networks.get('output_layer')])

        adam = Adam(lr=0.005)
        self.q_model.compile(loss='categorical_crossentropy', optimizer=adam,
                             metrics=['categorical_accuracy', 'categorical_crossentropy', 'mean_squared_error', 'acc'])

        print('model', self.q_model.summary())

    @staticmethod
    def give_dict_as_list(search_content, explore_dict):
        return [explore_dict.get('{}'.format(find)) for find in search_content]

    def train_the_model(self, hand_list, table_list, played_card_list, trumpf_list, target_list):

        input_list = []
        target_layer = []

        for i in range(len(hand_list)):
            input_list.append(create_input(hand_list[i], table_list[i], played_card_list[i], trumpf_list[i]))
            target_layer.append(create_target(target_list[i]))

        x = np.zeros((np.array(input_list).shape[0], INPUT_LAYER))
        y = np.zeros((np.array(target_layer).shape[0], OUTPUT_LAYER))

        print('x: ', x.shape)
        print('y: ', y.shape)

        x[:, :] = input_list
        y[:, :] = target_layer

        print("Input-Layer: {}".format(x))
        print("Output-Layer: {}".format(y))

        x = x.reshape((223, len(x), 6))
        y = y.reshape((1, len(y), 36))

        x_cards, x_trumpf = self.create_input_shapes(x, amount_card_sets)

        if self.game_counter % 500 == 0:
            self.q_model.fit(self.gen_fitting_dict(x_cards, x_trumpf, amount_card_sets), {'dense_46': y[0]},
                             validation_split=0.1, epochs=10, verbose=1, callbacks=[self.tb_callback])
        else:

            self.q_model.fit(self.gen_fitting_dict(x_cards, x_trumpf, amount_card_sets), {'dense_46': y[0]},
                             validation_split=0.1, epochs=10, verbose=1)
        self.game_counter += 1
        print("One Training-Part are finished!")

    @staticmethod
    def gen_fitting_dict(card_shape, trumpf_shape, amount_of_parts):
        x_fitting_dict = {}
        for i in range(amount_of_parts):
            x_fitting_dict['input_{}'.format(i + 1)] = card_shape[i]
        x_fitting_dict['input_38'] = trumpf_shape[0]
        return x_fitting_dict

    @staticmethod
    def gen_data_shape(input_shape, shape_index, array_index, shape_rows):
        gen_list = []
        for row in range(shape_rows):
            gen_list.append(input_shape[row + shape_index * 2][array_index])
        return gen_list

    def create_input_shapes(self, input_shape, card_sets_shapes):
        shape_rows = 6
        x_card_shape = np.ones((card_sets_shapes, len(input_shape[0]), shape_rows * len(input_shape[0][0])))
        x_trumpf_shape = np.ones((1, len(input_shape[0]), len(input_shape[0][0])))

        for dataset in range(len(input_shape[0])):
            for shape_index in range(card_sets_shapes):
                x_card_shape[shape_index][dataset] = np.concatenate((self.gen_data_shape(
                    input_shape, shape_index, dataset, shape_rows)))
            x_trumpf_shape[0][dataset] = np.array(input_shape[len(input_shape) - 1][dataset])

        return x_card_shape, x_trumpf_shape


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
        inputs[pos_player_on_table[x] + card.id] = 1

    if played_cards is None:
        return None
    if len(played_cards) > 0 and isinstance(played_cards[0], Card):
        print('The played card list has a wrong format!')
        return None
    else:
        for player_of_cards in range(len(played_cards)):
            if played_cards[player_of_cards] is None:
                break
            stich = 0
            for played_card in played_cards[player_of_cards]:
                inputs[pos_players_per_stich[stich][player_of_cards] + played_card.id] = 1
                stich += 1

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
