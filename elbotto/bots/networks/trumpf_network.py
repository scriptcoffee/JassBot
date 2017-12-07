import random
import numpy as np
from collections import deque
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from elbotto.bots.helpers import jass_helper


class TrumpfNetwork:
    INPUT_LAYER = 37
    FIRST_LAYER = 37
    OUTPUT_LAYER = 7

    REWARD_SCALING_FACTOR = 100
    LEARNING_RATE = 0.001

    def __init__(self):
        self.memory = deque(maxlen=50000)

        self.define_model()

    def define_model(self):
        self.input = Input(shape=(self.INPUT_LAYER,), name='trumpf_input')
        dense_1 = Dense(self.FIRST_LAYER, activation='relu', kernel_initializer='truncated_normal')(self.input)
        batch_norm_1 = BatchNormalization()(dense_1)
        self.dense_out = Dense(self.OUTPUT_LAYER, kernel_regularizer=l2(0.01), name='trumpf_output')(batch_norm_1)

        self.model = Model(inputs=self.input, outputs=self.dense_out)

        adam = Adam(lr=self.LEARNING_RATE)
        self.model.compile(optimizer=adam,
                           loss='categorical_crossentropy',
                           metrics=['mean_squared_error', 'categorical_accuracy', 'accuracy'])

    def model_choose_trumpf(self, hand_cards, geschoben):
        inputs = np.zeros((self.INPUT_LAYER,))

        for hand_card in hand_cards:
            inputs[hand_card.id] = 1
        inputs[self.INPUT_LAYER - 1] = geschoben

        inputs = np.reshape(inputs, (1, self.INPUT_LAYER))

        while True:
            q = self.model.predict(inputs)

            trumpf_nr = np.argmax(q)

            reward = jass_helper.evaluate_trumpf_choise(hand_cards, trumpf_nr, geschoben)

            self.memory.append((inputs, trumpf_nr, reward / self.REWARD_SCALING_FACTOR, 1))

            if not (geschoben and trumpf_nr == 6):
                return trumpf_nr

    def generate_states_targets(self, length):
        minibatch = random.sample(self.memory, length)

        states = np.zeros((length, self.INPUT_LAYER))
        targets = np.zeros((length, self.OUTPUT_LAYER))

        index = 0
        for state, action, reward, done in minibatch:
            target_f = self.model.predict(state)
            target_f[0][action] = reward

            states[index] = state
            targets[index] = target_f
            index += 1

        return states, targets
