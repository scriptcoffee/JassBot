import logging
import random
import keras
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.utils import np_utils

from elbotto.basebot import BaseBot, DEFAULT_TRUMPF

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
        card = self.game_strategy.choose_card(self.hand_cards, table_cards, self.game_type)
        return card

    def handle_stich(self, winner, round_points, total_points):
        self.game_strategy.stich_reward(round_points)

    def handle_game_finished(self):
        super(Bot, self).handle_game_finished()
        self.game_strategy.game_finished()


class PlayStrategy(object):

    def __init__(self):
        self.geschoben = False
        self.cardsAtTable = []

        self.gamma = 0.95
        self.epsilon = 0.6
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32

        self.reset_tmp_memory()

        self.memory = deque(maxlen=50000)

        self.q_model = self.define_model()

    def reset_tmp_memory(self):
        self.reward = None
        self.old_observation = None
        self.action = None

    @staticmethod
    def define_model():
        q_model = Sequential()
        q_model.add(Dense(38, input_shape=(42,), kernel_initializer='uniform'))
        q_model.add(keras.layers.normalization.BatchNormalization())
        q_model.add(Activation("relu"))
        q_model.add(Dense(36, kernel_regularizer=l2(0.01)))
        sgd = SGD(lr=0.005)
        q_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_squared_error'])

        return q_model

    def choose_trumpf(self, hand_cards):
        inputs = [0] * 36

        for c in hand_cards:
            inputs[c.id] = 1

        # CHALLENGE2017: Implement logic to chose game mode which is best suited to your handcards or schiaebae.
        # Consider that this decision ist quite crucial for your bot to be competitive
        # Use hearts as TRUMPF for now

        # if self.gschobe: nüme schiebe
        return DEFAULT_TRUMPF

    def choose_card(self, hand_cards, table_cards, game_type):

        card_to_play = self.model_choose_card(game_type, hand_cards, table_cards)

        if random.random() < self.epsilon:
            idx = random.randint(0, len(hand_cards)-1)
            card_to_play = hand_cards[idx]

        return card_to_play

    def card_rejected(self):
        self.reward = CARD_REJECTED_PENALTY

    def stich_reward(self, round_points):
        self.reward = round_points

    def game_finished(self):
        self.memory.append((self.old_observation, self.action, self.reward, None, 1))
        self.reset_tmp_memory()
        self.replay()

    def model_choose_card(self, game_type, hand_cards, table_cards):
        # 36 Inputs (one per card).
        # Status: 0 - no info, 1 - in hand, 2 - first card on table, 3 - second card on table, 4 - third card on table

        inputs = np.zeros((42,))

        for card in hand_cards:
            inputs[card.id] = 1

        for x in range(0, len(table_cards)):
            c = table_cards[x]
            c = card.create(c["number"], c["color"])
            inputs[c.id] = x + 2

        if game_type.mode == "TRUMPF":
            inputs[game_type.trumpf_color.value + 36] = 1

        elif game_type.mode == "OBEABE":
            inputs[40] = 1

        elif game_type.mode == "OBEABE":
            inputs[41] = 1

        i = np.reshape(inputs, (1, 42))

        if self.old_observation is not None and self.action is not None and self.reward is not None:
            self.memory.append((self.old_observation, self.action, self.reward, i, 0))

        q = self.q_model.predict(i)

        card_to_play = hand_cards[0]

        card_q = None
        for c in hand_cards:
            if card_q is None or card_q < q[0, c.id]:
                card_to_play = c
                card_q = q[0, c.id]

        self.old_observation = i
        self.action = card_to_play.id

        return card_to_play

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.q_model.predict(next_state)[0]))
            target_f = self.q_model.predict(state)
            target_f[0][action] = target
            history = self.q_model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay