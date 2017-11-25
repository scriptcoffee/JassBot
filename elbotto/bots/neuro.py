import logging
import random
import keras
import time
import numpy as np
import tensorflow as tf
from elbotto.messages import GameType
from elbotto.card import Card, Color
from keras import backend as k
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from datetime import datetime
from json import dump
from collections import deque
from keras.regularizers import l2
from keras.optimizers import SGD

from elbotto.basebot import BaseBot

logger = logging.getLogger(__name__)

CARD_REJECTED_PENALTY = -100


def save_weights(model, path):
    model.save_weights(path)
    return print("The weights of your model saved.")


def save_model(model, path, json=False):
    if json:
        model_json = model.to_json()
        with open(path, 'w') as f:
            dump(model_json, f)
        save_type = 'json'
    else:
        model.save(path)
        save_type = 'h5'
    return print("The model saved as " + save_type + ".")


def trumpf_hearts():
    return GameType("TRUMPF", Color.HEARTS.name)


def trumpf_diamonds():
    return GameType("TRUMPF", Color.DIAMONDS.name)


def trumpf_clubs():
    return GameType("TRUMPF", Color.CLUBS.name)


def trumpf_spades():
    return GameType("TRUMPF", Color.SPADES.name)


def obeabe():
    return GameType("OBEABE")


def undeufe():
    return GameType("UNDEUFE")


def shift():
    return GameType("SCHIEBE")


TRUMPF_DICT = {
    0: trumpf_hearts,
    1: trumpf_diamonds,
    2: trumpf_clubs,
    3: trumpf_spades,
    4: obeabe,
    5: undeufe,
    6: shift
}


class Bot(BaseBot):
    """
    Trivial bot using DEFAULT_TRUMPF and randomly returning a card available in the hand.
    This is a simple port of the original Java Script implementation
    """

    def __init__(self, server_address, name, chosen_team_index=0):
        super(Bot, self).__init__(server_address, name, chosen_team_index)
        self.game_strategy = PlayStrategy()

        self.played_cards = []

        self.start()

    def handle_request_trumpf(self):
        # CHALLENGE2017: Ask the brain which gameMode to choose
        return self.game_strategy.choose_trumpf(self.hand_cards)

    def handle_stich(self, winner, round_points, total_points, played_cards):
        won_stich = self.in_my_team(winner)
        self.played_cards.extend(played_cards)
        self.game_strategy.stich_reward(round_points)
        logger.debug("Stich: Won:%s, Winner: %s, Round points: %s, Total points: %s", won_stich, winner, round_points, total_points)

    def handle_reject_card(self, card):
        # CHALLENGE2017: When server sends this, you send an invalid card... this should never happen!
        # Server will send "REQUEST_CARD" after this once.
        # Make sure you choose a valid card or your bot will loose the game
        logger.debug(" ######   SERVER REJECTED CARD   #######")
        logger.debug("Rejected card: %s", card)
        logger.debug("Hand Cards: %s", self.hand_cards)
        logger.debug("cardsAtTable %s", self.game_strategy.cardsAtTable)
        logger.debug("Gametype: %s", self.game_type)
        self.game_strategy.card_rejected()

    def handle_request_card(self, table_cards):
        # CHALLENGE2017: Ask the brain which card to choose
        return self.game_strategy.choose_card(self.hand_cards, table_cards, self.game_type, self.played_cards)

    def handle_game_finished(self):
        self.played_cards = []
        super(Bot, self).handle_game_finished()
        self.game_strategy.game_finished()


class PlayStrategy:
    TRUMPF_INPUT_LAYER = 37
    TRUMPF_FIRST_LAYER = 37
    TRUMPF_OUTPUT_LAYER = 7

    INPUT_LAYER = 186
    FIRST_LAYER = 50
    OUTPUT_LAYER = 36

    def __init__(self, save_models=True):
        self.geschoben = False
        self.cardsAtTable = []
        self.game_counter = 1

        self.gamma = 0.95
        self.epsilon = 0.6
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 16

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        sess = tf.Session(config=config)
        k.set_session(sess)

        self.reset_tmp_memory()

        self.trumpf_memory = deque(maxlen=50000)
        self.round_memory = deque(maxlen=50)
        self.game_memory = deque(maxlen=50000)

        self.define_models()
        if save_models:
            self.save_weights_and_models()
        self.writer = tf.summary.FileWriter('./logs/')
        self.step = 0
        self.time = time.time()
        self.tb_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=5, batch_size=32, write_graph=False,
                                    write_grads=True, write_images=False, embeddings_freq=0,
                                    embeddings_layer_names=None, embeddings_metadata=None)

    def define_models(self):
        trumpf_input = Input(shape=(self.TRUMPF_INPUT_LAYER,), name='trumpf_input')
        trumpf_dense_1 = Dense(self.TRUMPF_FIRST_LAYER, activation='relu', kernel_initializer='truncated_normal')(trumpf_input)
        trumpf_batch_norm_1 = BatchNormalization()(trumpf_dense_1)
        trumpf_dense_out = Dense(self.TRUMPF_OUTPUT_LAYER, kernel_regularizer=l2(0.01), name='trumpf_output')(trumpf_batch_norm_1)

        self.trumpf_model = Model(inputs=trumpf_input, outputs=trumpf_dense_out)

        game_imput = Input(shape=(self.INPUT_LAYER,), name='game_input')
        game_dense_1 = Dense(self.FIRST_LAYER, activation='relu', kernel_initializer='truncated_normal')(game_imput)
        game_batch_norm_1 = BatchNormalization()(game_dense_1)
        game_dense_out = Dense(self.OUTPUT_LAYER, kernel_regularizer=l2(0.01), name='game_output')(game_batch_norm_1)

        self.game_model = Model(inputs=game_imput, outputs=game_dense_out)

        self.combined_model = Model(inputs=[trumpf_input, game_imput], outputs=[trumpf_dense_out, game_dense_out])
        sgd = SGD(lr=0.001)
        self.trumpf_model.compile(optimizer=sgd,
                      loss='mean_squared_error',
                      metrics=['mean_squared_error', 'accuracy'])
        self.game_model.compile(optimizer=sgd,
                      loss='mean_squared_error',
                      metrics=['mean_squared_error', 'accuracy'])
        self.combined_model.compile(optimizer=sgd,
                      loss='mean_squared_error',
                      metrics=['mean_squared_error', 'accuracy'])

    def reset_tmp_memory(self):
        self.trumpf_reward = None
        self.trumpf_observation = None
        self.trumpf_action = None

        self.game_reward = None
        self.game_old_observation = None
        self.game_action = None

    def choose_trumpf(self, hand_cards):
        inputs = np.zeros((self.TRUMPF_INPUT_LAYER,))

        for hand_card in hand_cards:
            inputs[hand_card.id] = 1
        inputs[self.TRUMPF_INPUT_LAYER - 1] = self.geschoben

        inputs = np.reshape(inputs, (1, self.TRUMPF_INPUT_LAYER))

        q = self.trumpf_model.predict(inputs)

        trumpf_nr = np.argmax(q)

        trumpf = TRUMPF_DICT[trumpf_nr]()

        trumpf_reward = evaluate_trumpf_choise(hand_cards, trumpf_nr, self.geschoben)

        self.trumpf_memory.append((inputs, trumpf_nr, trumpf_reward/100, 1))

        return trumpf

    def replay_trumpf(self, length):
        minibatch = random.sample(self.trumpf_memory, length)

        states = np.zeros((length, self.TRUMPF_INPUT_LAYER))
        targets = np.zeros((length, self.TRUMPF_OUTPUT_LAYER))

        index = 0
        for state, action, reward, done in minibatch:
            target_f = self.trumpf_model.predict(state)
            target_f[0][action] = reward

            states[index] = state
            targets[index] = target_f
            index += 1

        return states, targets

    def choose_card(self, hand_cards, table_cards, game_type, played_cards):

        inputs = self.prepare_game_input(game_type, hand_cards, table_cards, played_cards)

        card_to_play = self.model_choose_card(inputs, hand_cards)

        if random.random() < self.epsilon:
            idx = random.randint(0, len(hand_cards)-1)
            card_to_play = hand_cards[idx]

        return card_to_play

    def card_rejected(self):
        self.game_reward = CARD_REJECTED_PENALTY / 100

    def stich_reward(self, round_points):
        self.game_reward = round_points / 100

    def game_finished(self):
        self.round_memory.append((self.game_old_observation, self.game_action, self.game_reward, 1))
        self.game_memory.append(list(self.round_memory))
        self.round_memory.clear()
        self.reset_tmp_memory()
        self.fit_models()
        if (self.game_counter % 1000) == 0:
            self.save_weights_and_models()
        self.game_counter += 1

    def fit_models(self):
        if len(self.game_memory) < self.batch_size:
            return 0
        game_states, game_targets = self.replay_games()

        if len(self.trumpf_memory) < len(game_states):
            return 0
        trumpf_states, trumpf_targets = self.replay_trumpf(len(game_states))

        if (self.game_counter % 100) == 0:
            self.combined_model.fit(x={'trumpf_input': trumpf_states, 'game_input': game_states},
                                y={'trumpf_output': trumpf_targets, 'game_output': game_targets},
                                epochs=5,
                                verbose=0,
                                validation_split=0.2,
                                callbacks=[self.tb_callback])
        else:
            self.combined_model.fit(x={'trumpf_input': trumpf_states, 'game_input': game_states},
                                y={'trumpf_output': trumpf_targets, 'game_output': game_targets},
                                epochs=5,
                                verbose=0,
                                validation_split=0.2)

        print(time.time() - self.time)
        self.time = time.time()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_weights_and_models(self):
        file_addition = str(self.game_counter) + datetime.now().strftime("__%Y-%m-%d_%H%M%S")
        save_model(self.game_model, "./logs/config/game_network_model_" + file_addition + ".h5")
        save_model(self.game_model, "./logs/config/game_network_model_" + file_addition + ".json", True)
        save_weights(self.game_model, "./logs/config/game_network_weights_" + file_addition + ".h5")

        save_model(self.trumpf_model, "./logs/config/trumpf_network_model_" + file_addition + ".h5")
        save_model(self.trumpf_model, "./logs/config/trumpf_network_model_" + file_addition + ".json", True)
        save_weights(self.trumpf_model, "./logs/config/trumpf_network_weights_" + file_addition + ".h5")

    def model_choose_card(self, inputs, hand_cards):

        if self.game_old_observation is not None and self.game_action is not None and self.game_reward is not None:
            self.round_memory.append((self.game_old_observation, self.game_action, self.game_reward, 0))

        q = self.game_model.predict(inputs)

        card_to_play = hand_cards[0]
        card_q = None
        for hand_card in hand_cards:
            if card_q is None or card_q < q[0, hand_card.id]:
                card_to_play = hand_card
                card_q = q[0, hand_card.id]

        card_to_play_rank = 0
        for prob in q[0]:
            if prob > card_q:
                card_to_play_rank += 1

        if (self.step % 1000) == 0:
            summary = tf.Summary(value=[tf.Summary.Value(tag="card_accuracy", simple_value=card_to_play_rank)])
            self.writer.add_summary(summary, self.step)
        self.step += 1

        self.game_old_observation = inputs
        self.game_action = card_to_play.id

        return card_to_play

    def prepare_game_input(self, game_type, hand_cards, table_cards, played_cards):
        trumpf_offset = self.INPUT_LAYER - 6
        inputs = np.zeros((self.INPUT_LAYER,))

        for hand_card in hand_cards:
            inputs[hand_card.id] = 1

        for i in range(len(table_cards)):
            table_card = table_cards[i]
            table_card = Card.create(table_card["number"], table_card["color"])
            input_index = (i + 1) * 36 + table_card.id
            inputs[input_index] = 1

        for played_card in played_cards:
            input_index = 4 * 36 + played_card.id
            inputs[input_index] = 1

        if game_type.mode == "TRUMPF":
            inputs[game_type.trumpf_color.value + trumpf_offset] = 1
        elif game_type.mode == "OBEABE":
            inputs[trumpf_offset + 4] = 1
        elif game_type.mode == "UNDEUFE":
            inputs[trumpf_offset + 5] = 1

        inputs = np.reshape(inputs, (1, self.INPUT_LAYER))

        return inputs

    def replay_games(self):
        minibatch = random.sample(self.game_memory, self.batch_size)

        states = []
        targets = []
        for game_round in minibatch:
            td_points = 0
            index = 0
            for state, action, reward, done in reversed(game_round):
                target = reward
                if index >= 1:
                    target += td_points / index

                td_points = self.gamma * (np.amax(self.game_model.predict(state)) + td_points)

                target_f = self.game_model.predict(state)
                target_f[0][action] = target

                states.append(np.squeeze(state, axis=0))
                targets.append(np.squeeze(target_f, axis=0))
                index += 1

        return np.array(states), np.array(targets)


def evaluate_trumpf_choise(hand_cards, trumpf_chosen, geschoben):
    guaranteed_stich_reward_color = 4
    guaranteed_stich_reward_nocolor = 11
    shift_score_threshold = 55
    no_shift_penalty = -30

    cards_per_color = {
        0: [],
        1: [],
        2: [],
        3: []}

    score_per_trumpf = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0}

    trumpf_card_score = {
        6: 10,
        7: 10,
        8: 10,
        9: 15,
        10: 10,
        11: 30,
        12: 10,
        13: 10,
        14: 12}

    for card in hand_cards:
        score_per_trumpf[card.color.value] += trumpf_card_score[card.number]
        cards_per_color[card.color.value].append(card)

    for trumpf_color in Color:
        for color in Color:
            if color.value != trumpf_color.value:
                score_per_trumpf[trumpf_color.value] += calc_guaranteed_stich_score(cards_per_color[color.value], True) * guaranteed_stich_reward_color
        score_per_trumpf[4] += calc_guaranteed_stich_score(cards_per_color[trumpf_color.value], True) * guaranteed_stich_reward_nocolor
        score_per_trumpf[5] += calc_guaranteed_stich_score(cards_per_color[trumpf_color.value], False) * guaranteed_stich_reward_nocolor

    max_score = np.amax(list(score_per_trumpf.values()))

    score = score_per_trumpf[trumpf_chosen] - max_score
    if max_score < shift_score_threshold and not geschoben:
        if trumpf_chosen == 6:
            score = 0
        else:
            score = no_shift_penalty

    return score


def calc_guaranteed_stich_score(cards, obenabe):
    stichs = 0
    if obenabe:
        highest_card = 14
    else:
        highest_card = 6

    cards.sort(key=lambda x: x.id, reverse=obenabe)

    for card in cards:
        if card.number == highest_card:
            stichs += 1
            if obenabe:
                highest_card -= 1
            else:
                highest_card += 1
        else:
            break

    return stichs
