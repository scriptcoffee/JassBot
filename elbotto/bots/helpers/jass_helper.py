import numpy as np
from elbotto.card import Card, Color
from elbotto.messages import GameType


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
