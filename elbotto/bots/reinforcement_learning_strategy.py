import logging
import random
import keras
import numpy as np
import tensorflow as tf
from elbotto.bots.helpers import keras_helper, jass_helper
from keras import backend as k
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from datetime import datetime
from collections import deque
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

logger = logging.getLogger(__name__)

CARD_REJECTED_PENALTY = -100

class PlayStrategy:
    TRUMPF_INPUT_LAYER = 37
    TRUMPF_FIRST_LAYER = 37
    TRUMPF_OUTPUT_LAYER = 7

    INPUT_LAYER = 186
    FIRST_LAYER = 50
    OUTPUT_LAYER = 36

    MATCH_REWARD = 100
    REWARD_SCALING_FACTOR = 100

    def __init__(self, save_models=True):
        self.game_counter = 1

        self.gamma = 0.95
        self.epsilon = 0.6
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 16

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
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
        adam = Adam(lr=0.001)
        self.trumpf_model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['mean_squared_error', 'categorical_accuracy', 'accuracy'])
        self.game_model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['mean_squared_error', 'categorical_accuracy', 'accuracy'])
        self.combined_model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['mean_squared_error', 'categorical_accuracy', 'accuracy'])

    def reset_tmp_memory(self):
        self.trumpf_reward = None
        self.trumpf_observation = None
        self.trumpf_action = None

        self.game_reward = None
        self.game_old_observation = None
        self.game_action = None

    def all_round_parameters_set(self):
        return self.game_old_observation is not None \
               and self.game_action is not None \
               and self.game_reward is not None

    def choose_trumpf(self, hand_cards, geschoben):
        inputs = np.zeros((self.TRUMPF_INPUT_LAYER,))

        for hand_card in hand_cards:
            inputs[hand_card.id] = 1
        inputs[self.TRUMPF_INPUT_LAYER - 1] = geschoben

        inputs = np.reshape(inputs, (1, self.TRUMPF_INPUT_LAYER))

        while True:
            q = self.trumpf_model.predict(inputs)

            trumpf_nr = np.argmax(q)

            if random.random() < self.epsilon:
                trumpf_nr = random.randint(0, self.TRUMPF_OUTPUT_LAYER - 2)

            trumpf = jass_helper.TRUMPF_DICT[trumpf_nr]()

            trumpf_reward = jass_helper.evaluate_trumpf_choise(hand_cards, trumpf_nr, geschoben)

            self.trumpf_memory.append((inputs, trumpf_nr, trumpf_reward / self.REWARD_SCALING_FACTOR, 1))

            if not (geschoben and trumpf_nr == 6):
                return trumpf

    def choose_card(self, hand_cards, table_cards, game_type, played_cards):
        inputs = keras_helper.prepare_game_input(self.INPUT_LAYER, game_type, hand_cards, table_cards, played_cards)

        card_to_play = self.model_choose_card(inputs, hand_cards)

        if random.random() < self.epsilon:
            idx = random.randint(0, len(hand_cards)-1)
            card_to_play = hand_cards[idx]

        return card_to_play

    def model_choose_card(self, inputs, hand_cards):
        if self.all_round_parameters_set():
            self.round_memory.append((self.game_old_observation, self.game_action, self.game_reward, 0))

        q = self.game_model.predict(inputs)

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

        self.game_old_observation = inputs
        self.game_action = card_to_play.id

        return card_to_play

    def card_rejected(self):
        self.game_reward = CARD_REJECTED_PENALTY / self.REWARD_SCALING_FACTOR

    def stich_reward(self, round_points):
        self.game_reward = round_points / self.REWARD_SCALING_FACTOR

    def game_finished(self, isMatch):
        self.round_memory.append((self.game_old_observation, self.game_action, self.game_reward, 1))
        if isMatch:
            round = []
            for stich in self.round_memory:
                stich = list(stich)
                stich[2] += (self.MATCH_REWARD / 9) / self.REWARD_SCALING_FACTOR
                round.append(stich)
            self.game_memory.append(round)
        else:
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
        game_states, game_targets = self.generate_game_states_targets()

        if len(self.trumpf_memory) < len(game_states):
            return 0
        trumpf_states, trumpf_targets = self.generate_trumpf_states_targets(len(game_states))

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

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def generate_trumpf_states_targets(self, length):
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

    def generate_game_states_targets(self):
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

                target_f = self.game_model.predict(state)
                td_points = self.gamma * (np.amax(target_f) + td_points)

                target_f[0][action] = target

                states.append(np.squeeze(state, axis=0))
                targets.append(np.squeeze(target_f, axis=0))
                index += 1

        return np.array(states), np.array(targets)

    def save_weights_and_models(self):
        log_dir = "./logs/config/"

        game_model_name = "game_network"
        trumpf_model_name = "trumpf_network"

        file_addition = "_" + str(self.game_counter) + datetime.now().strftime("__%Y-%m-%d_%H%M%S")

        keras_helper.save_model(self.game_model, "{}{}_model{}.h5".format(log_dir, game_model_name, file_addition))
        keras_helper.save_model(self.game_model, "{}{}_model{}.json".format(log_dir, game_model_name, file_addition), True)
        keras_helper.save_weights(self.game_model, "{}{}_weights{}.h5".format(log_dir, game_model_name, file_addition))

        keras_helper.save_model(self.trumpf_model, "{}{}_model{}.h5".format(log_dir, trumpf_model_name, file_addition))
        keras_helper.save_model(self.trumpf_model, "{}{}_model{}.json".format(log_dir, trumpf_model_name, file_addition), True)
        keras_helper.save_weights(self.trumpf_model, "{}{}_weights{}.h5".format(log_dir, trumpf_model_name, file_addition))
