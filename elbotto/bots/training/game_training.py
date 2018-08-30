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

INPUT_LAYER = 1338
FIRST_LAYER = 4000
SECOND_LAYER = 12000
THIRD_LAYER = 36000
FOURTH_LAYER = 4000
FIFTH_LAYER = 440
OUTPUT_LAYER = 36

CARD_SET = 36

pos_hand_cards = 0
pos_player_on_table = [1 * CARD_SET, 2 * CARD_SET, 3 * CARD_SET, 4 * CARD_SET]
# pos_table = 1 * CARD_SET


def generate_player_per_stich(start_range, end_range):
    player_collector = []
    pos_players_per_stich = []

    for player in range(start_range, end_range):
        player_collector.append(player * CARD_SET)
        if len(player_collector) == 4 or player == end_range:
            pos_players_per_stich.append(player_collector)
            player_collector = []

    return pos_players_per_stich


pos_players_per_stich = generate_player_per_stich(5,37)

pos_trumpf = INPUT_LAYER - 6  # Alternativ calculation: 6 * CARD_SET


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
        self.q_model.add(Dense(FOURTH_LAYER, activation='relu', kernel_initializer='uniform'))
        self.q_model.add(BatchNormalization())
        self.q_model.add(Dense(FIFTH_LAYER, activation='relu', kernel_initializer='uniform'))
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


