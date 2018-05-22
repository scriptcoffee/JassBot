import os
import keras
from elbotto.bots.training.card_parser import create_card
from elbotto.bots.training.trumpf_converter import trumpf_converter
from elbotto.bots.training.trumpf_training import create_input


def is_none(input_object):
    if input_object is None:
        return []
    if isinstance(input_object, list):
        for i in input_object:
            is_none(i)
    return input_object


def get_model(model):
    if model is None:
        print("Give a file with a valid model.")
        return None
    else:
        return create_model_from_file(model)


def create_model_from_file(model_path):
    os.chdir(os.path.dirname(__file__))
    q_model = keras.models.load_model(model_path)
    return q_model


def fillup_card_list(cards):
    cards = is_none(cards)
    if not isinstance(cards, list):
        return None
    card_list = []
    if len(cards) > 0:
        for card in cards:
            card_list.append(create_card(card))
    return card_list


def create_test_matrix(hand_cards, pushed_from_partner=False):
    if isinstance(pushed_from_partner, bool) and pushed_from_partner:
        shift = trumpf_converter(6)
    else:
        shift = trumpf_converter(0)

    hand_card_list = fillup_card_list(hand_cards)

    return create_input(hand_card_list, shift)