import logging
import random
import keras
import tensorflow as tf
from keras import backend as k
from keras.models import Model
from keras.optimizers import Adam
from elbotto.bots.helpers import model_helper, jass_helper
from elbotto.bots.networks.game_network import GameNetwork
from elbotto.bots.networks.trumpf_network import TrumpfNetwork

logger = logging.getLogger(__name__)


class PlayStrategy:
    NR_OF_TRUMPFS = 6

    def __init__(self, save_models=True):
        self.game_counter = 1

        self.epsilon = 0.05
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 16
        self.learning_rate = 0.001

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        k.set_session(sess)

        self.trumpf_network = TrumpfNetwork()
        self.game_network = GameNetwork()

        self.define_models()
        if save_models:
            self.save_weights_and_models()
        self.step = 0

        self.tb_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=5, batch_size=32, write_graph=False,
                                    write_grads=True, write_images=False, embeddings_freq=0,
                                    embeddings_layer_names=None, embeddings_metadata=None)

    def define_models(self):
        self.combined_model = Model(inputs=[self.trumpf_network.input, self.game_network.input],
                                    outputs=[self.trumpf_network.dense_out, self.game_network.dense_out])
        adam = Adam(lr=self.learning_rate)
        self.combined_model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['mean_squared_error', 'categorical_accuracy', 'accuracy'])

    def choose_trumpf(self, hand_cards, geschoben):
        trumpf_nr = self.trumpf_network.model_choose_trumpf(hand_cards, geschoben)

        if random.random() < self.epsilon:
            trumpf_nr = random.randint(0, self.NR_OF_TRUMPFS)

        return jass_helper.TRUMPF_DICT[trumpf_nr]()

    def choose_card(self, hand_cards, table_cards, game_type, played_cards):
        card_to_play = self.game_network.model_choose_card(game_type, hand_cards, table_cards, played_cards)

        if random.random() < self.epsilon:
            idx = random.randint(0, len(hand_cards)-1)
            card_to_play = hand_cards[idx]

        return card_to_play

    def card_rejected(self):
        self.game_network.card_rejected()

    def stich_reward(self, round_points):
        self.game_network.stich_reward(round_points)

    def game_finished(self, is_match):
        self.game_network.save_round(is_match)
        self.game_network.reset_tmp_memory()

        self.fit_models()
        if (self.game_counter % 1000) == 0:
            self.save_weights_and_models()

        self.game_counter += 1

    def fit_models(self):
        if len(self.game_network.memory) < self.batch_size:
            return 0
        game_states, game_targets = self.game_network.generate_states_targets(self.batch_size)

        if len(self.trumpf_network.memory) < len(game_states):
            return 0
        trumpf_states, trumpf_targets = self.trumpf_network.generate_states_targets(len(game_states))

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

    def save_weights_and_models(self):
        log_dir = "./logs/config/"

        model_helper.save_weights_and_model(self.trumpf_network.model, self.game_counter, log_dir, "trumpf_network")
        model_helper.save_weights_and_model(self.game_network.model, self.game_counter, log_dir, "game_network")
