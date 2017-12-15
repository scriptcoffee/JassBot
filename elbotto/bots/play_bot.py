import logging
import random
import numpy as np
import tensorflow as tf
from elbotto.messages import GameType
from elbotto.card import Card, Color
from keras import backend as k
from keras.models import load_model

logger = logging.getLogger(__name__)


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


class PlayStrategy:
    TRUMPF_INPUT_LAYER = 37
    TRUMPF_OUTPUT_LAYER = 7

    INPUT_LAYER = 186

    def __init__(self):
        self.epsilon = 0.05

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        sess = tf.Session(config=config)
        k.set_session(sess)

        self.trumpf_model = load_model("models/trumpf_network_model.h5")
        self.game_model = load_model("models/game_network_model.h5")

        self.writer = tf.summary.FileWriter('./logs/')
        self.step = 0

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

            trumpf = TRUMPF_DICT[trumpf_nr]()

            if not (geschoben and trumpf_nr == 6):
                return trumpf

    def choose_card(self, hand_cards, table_cards, game_type, played_cards):

        inputs = prepare_game_input(self.INPUT_LAYER, game_type, hand_cards, table_cards, played_cards)

        card_to_play = self.model_choose_card(inputs, hand_cards)

        if random.random() < self.epsilon:
            idx = random.randint(0, len(hand_cards)-1)
            card_to_play = hand_cards[idx]

        return card_to_play

    def model_choose_card(self, inputs, hand_cards):
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

        return card_to_play

    def card_rejected(self):
        pass

    def stich_reward(self, round_points):
        pass

    def game_finished(self, is_match):
        pass


def prepare_game_input(input_layer_size, game_type, hand_cards, table_cards, played_cards):
    trumpf_offset = input_layer_size - 6
    inputs = np.zeros((input_layer_size,))

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

    inputs = np.reshape(inputs, (1, input_layer_size))

    return inputs
