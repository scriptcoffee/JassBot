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


class Bot(BaseBot):
    """
    Trivial bot using DEFAULT_TRUMPF and randomly returning a card available in the hand.
    This is a simple port of the original Java Script implementation
    """

    def __init__(self, server_address, name, chosen_team_index=0):
        super(Bot, self).__init__(server_address, name, chosen_team_index)
        self.game_strategy = PlayStrategy()

        self.start()


    def handle_request_trumpf(self):
        # CHALLENGE2017: Ask the brain which gameMode to choose
        return self.game_strategy.choose_trumpf(self.hand_cards)

    def handle_stich(self, winner, round_points, total_points):
        won_stich = self.in_my_team(winner)
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
        return self.game_strategy.choose_card(self.hand_cards, table_cards, self.game_type)

    def handle_game_finished(self, current_game_points, won_stich_in_game):
        super(Bot, self).handle_game_finished(current_game_points, won_stich_in_game)
        self.game_strategy.game_finished(current_game_points, won_stich_in_game)


class PlayStrategy(object):
    TRUMPF_INPUT_LAYER = 36
    TRUMPF_FIRST_LAYER = 36
    TRUMPF_OUTPUT_LAYER = 7

    INPUT_LAYER = 150
    FIRST_LAYER = 50
    OUTPUT_LAYER = 36

    def __init__(self):
        self.geschoben = False
        self.cardsAtTable = []
        self.game_counter = 1

        self.gamma = 0.95
        self.epsilon = 0.6
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.trumpf_batch_size = 16
        self.batch_size = 16

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        sess = tf.Session(config=config)
        k.set_session(sess)

        self.reset_tmp_memory()

        self.trumpf_memory = deque(maxlen=50000)
        self.game_memory = deque(maxlen=50000)

        self.define_models()
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

    @staticmethod
    def save_weights(model, path):
        model.save_weights(path)
        return print("The weights of your model saved.")

    @staticmethod
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

    def choose_trumpf(self, hand_cards):
        inputs = np.zeros((self.TRUMPF_INPUT_LAYER,))

        for c in hand_cards:
            inputs[c.id] = 1

        i = np.reshape(inputs, (1, self.TRUMPF_INPUT_LAYER))

        q = self.trumpf_model.predict(i)

        trumpf_nr = np.argmax(q)

        if trumpf_nr == 0:
            trumpf = GameType("TRUMPF", Color.HEARTS.name)
        elif trumpf_nr == 1:
            trumpf = GameType("TRUMPF", Color.DIAMONDS.name)
        elif trumpf_nr == 2:
            trumpf = GameType("TRUMPF", Color.CLUBS.name)
        elif trumpf_nr == 3:
            trumpf = GameType("TRUMPF", Color.SPADES.name)
        elif trumpf_nr == 4:
            trumpf = GameType("OBEABE")
        else:
            trumpf = GameType("UNDEUFE")

        self.trumpf_observation = i
        self.trumpf_action = trumpf_nr

        # if self.gschobe: nüme schiebe
        return trumpf

    def replay_trumpf(self):
        minibatch = random.sample(self.trumpf_memory, self.trumpf_batch_size)

        states = np.zeros((self.trumpf_batch_size, self.TRUMPF_INPUT_LAYER))
        targets = np.zeros((self.trumpf_batch_size, self.TRUMPF_OUTPUT_LAYER))

        index = 0
        for state, action, reward, done in minibatch:
            target_f = self.trumpf_model.predict(state)
            target_f[0][action] = reward

            states[index] = state
            targets[index] = target_f
            index += 1

        return states, targets

    def choose_card(self, hand_cards, table_cards, game_type):

        card_to_play = self.model_choose_card(game_type, hand_cards, table_cards)

        if random.random() < self.epsilon:
            idx = random.randint(0, len(hand_cards)-1)
            card_to_play = hand_cards[idx]

        return card_to_play

    def card_rejected(self):
        self.game_reward = CARD_REJECTED_PENALTY / 100

    def stich_reward(self, round_points):
        self.game_reward = round_points / 100

    def game_finished(self, current_game_points, won_stich_in_game):
        if self.trumpf_observation is not None and self.trumpf_action is not None:
            self.trumpf_memory.append((self.trumpf_observation, self.trumpf_action, (won_stich_in_game.count(1)/10), 1))
        self.game_memory.append((self.game_old_observation, self.game_action, self.game_reward, None, 1))
        self.reset_tmp_memory()
        self.fit_models()
        if (self.game_counter % 1000) == 0:
            self.save_weights_and_models()
        self.game_counter += 1

    def fit_models(self):
        if len(self.trumpf_memory) < self.trumpf_batch_size or len(self.game_memory) < self.batch_size:
            return 0

        trumpf_states, trumpf_targets = self.replay_trumpf()
        game_states, game_targets = self.replay_games()

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
        self.save_model(self.game_model, "./logs/config/game_network_model_" + file_addition + ".h5")
        self.save_model(self.game_model, "./logs/config/game_network_model_" + file_addition + ".json", True)
        self.save_weights(self.game_model, "./logs/config/game_network_weights_" + file_addition + ".h5")

        self.save_model(self.trumpf_model, "./logs/config/trumpf_network_model_" + file_addition + ".h5")
        self.save_model(self.trumpf_model, "./logs/config/trumpf_network_model_" + file_addition + ".json", True)
        self.save_weights(self.trumpf_model, "./logs/config/trumpf_network_weights_" + file_addition + ".h5")

    def model_choose_card(self, game_type, hand_cards, table_cards):
        # 4 x 36 Inputs (one per card per status).
        #   0 -  35 : cards on hand
        #  36 -  71 : first card played
        #  72 - 107 : second card played
        # 108 - 143 : third card played

        trumpf_offset = self.INPUT_LAYER - 6

        inputs = np.zeros((self.INPUT_LAYER,))
        for c in hand_cards:
            inputs[c.id] = 1

        for x in range(0, len(table_cards)):
            c = table_cards[x]
            c = Card.create(c["number"], c["color"])
            input_index = (x+1) * 36 + c.id
            inputs[input_index] = 1

        if game_type.mode == "TRUMPF":
            inputs[game_type.trumpf_color.value + trumpf_offset] = 1
        elif game_type.mode == "OBEABE":
            inputs[trumpf_offset + 4] = 1
        elif game_type.mode == "UNDEUFE":
            inputs[trumpf_offset + 5] = 1

        i = np.reshape(inputs, (1, self.INPUT_LAYER))

        if self.game_old_observation is not None and self.game_action is not None and self.game_reward is not None:
            self.game_memory.append((self.game_old_observation, self.game_action, self.game_reward, i, 0))

        q = self.game_model.predict(i)

        card_to_play = hand_cards[0]

        card_q = None
        for c in hand_cards:
            if card_q is None or card_q < q[0, c.id]:
                card_to_play = c
                card_q = q[0, c.id]

        card_to_play_rank = 0
        for prob in q[0]:
            if prob > card_q:
                card_to_play_rank += 1

        if (self.step % 1000) == 0:
            summary = tf.Summary(value=[tf.Summary.Value(tag="card_accuracy", simple_value=card_to_play_rank)])
            self.writer.add_summary(summary, self.step)
        self.step += 1

        self.game_old_observation = i
        self.game_action = card_to_play.id

        return card_to_play

    def replay_games(self):
        minibatch = random.sample(self.game_memory, self.batch_size)

        states = np.zeros((self.batch_size, self.INPUT_LAYER))
        targets = np.zeros((self.batch_size, self.OUTPUT_LAYER))
        index = 0
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.game_model.predict(next_state)[0]))
            target_f = self.game_model.predict(state)
            target_f[0][action] = target

            states[index] = state
            targets[index] = target_f
            index += 1

        return states, targets
