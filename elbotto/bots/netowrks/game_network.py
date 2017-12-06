import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from elbotto.bots.helpers import keras_helper


class GameNetwork:
    INPUT_LAYER = 186
    FIRST_LAYER = 50
    OUTPUT_LAYER = 36

    MATCH_REWARD = 100
    CARD_REJECTED_PENALTY = -100
    REWARD_SCALING_FACTOR = 100

    def __init__(self):
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.step = 0

        self.round_memory = deque(maxlen=50)
        self.memory = deque(maxlen=50000)

        self.reset_tmp_memory()

        self.writer = tf.summary.FileWriter('./logs/')

    def reset_tmp_memory(self):
        self.reward = None
        self.old_observation = None
        self.action = None

    def all_round_parameters_set(self):
        return self.old_observation is not None \
               and self.action is not None \
               and self.reward is not None

    def define_model(self):
        self.input = Input(shape=(self.INPUT_LAYER,), name='game_input')
        dense_1 = Dense(self.FIRST_LAYER, activation='relu', kernel_initializer='truncated_normal')(self.input)
        batch_norm_1 = BatchNormalization()(dense_1)
        self.dense_out = Dense(self.OUTPUT_LAYER, kernel_regularizer=l2(0.01), name='game_output')(batch_norm_1)

        self.model = Model(inputs=self.input, outputs=self.dense_out)

        adam = Adam(lr=self.learning_rate)
        self.model.compile(optimizer=adam,
                           loss='categorical_crossentropy',
                           metrics=['mean_squared_error', 'categorical_accuracy', 'accuracy'])

    def model_choose_card(self, game_type, hand_cards, table_cards, played_cards):
        inputs = keras_helper.prepare_game_input(self.INPUT_LAYER, game_type, hand_cards, table_cards, played_cards)
        if self.all_round_parameters_set():
            self.round_memory.append((self.old_observation, self.action, self.reward, 0))

        q = self.model.predict(inputs)

        card_to_play = hand_cards[0]
        card_q = None
        for hand_card in hand_cards:
            if card_q is None or card_q < q[0, hand_card.id]:
                card_to_play = hand_card
                card_q = q[0, hand_card.id]

        if (self.step % 1000) == 0:
            card_to_play_rank = len([prob for prob in q[0] if prob > card_q])
            summary = tf.Summary(value=[tf.Summary.Value(tag="card_accuracy", simple_value=card_to_play_rank)])
            self.writer.add_summary(summary, self.step)
        self.step += 1

        self.old_observation = inputs
        self.action = card_to_play.id

        return card_to_play

    def card_rejected(self):
        self.reward = self.CARD_REJECTED_PENALTY / self.REWARD_SCALING_FACTOR

    def stich_reward(self, round_points):
        self.reward = round_points / self.REWARD_SCALING_FACTOR

    def save_round(self, is_match):
        self.round_memory.append((self.old_observation, self.action, self.reward, 1))
        if is_match:
            round = []
            for stich in self.round_memory:
                stich = list(stich)
                stich[2] += (self.MATCH_REWARD / 9) / self.REWARD_SCALING_FACTOR
                round.append(stich)
            self.memory.append(round)
        else:
            self.memory.append(list(self.round_memory))
        self.round_memory.clear()

    def generate_states_targets(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        states = []
        targets = []
        for game_round in minibatch:
            td_points = 0
            index = 0
            for state, action, reward, done in reversed(game_round):
                target = reward
                if index >= 1:
                    target += td_points / index

                target_f = self.model.predict(state)
                td_points = self.gamma * (np.amax(target_f) + td_points)

                target_f[0][action] = target

                states.append(np.squeeze(state, axis=0))
                targets.append(np.squeeze(target_f, axis=0))
                index += 1

        return np.array(states), np.array(targets)
