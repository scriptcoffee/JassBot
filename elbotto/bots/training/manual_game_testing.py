import numpy as np
from elbotto.bots.training.manual_testing import is_none, get_model, fillup_card_list
from elbotto.bots.training.trumpf_converter import trumpf_converter
from elbotto.bots.training.game_training import create_input
from elbotto.card import Card


def create_test_matrix(hand_cards, table_cards, played_cards, game_type):
    hand_card_list = is_none(fillup_card_list(hand_cards))
    table_card_list = is_none(fillup_card_list(table_cards))
    played_card_list = []
    if len(played_cards) == 4 and isinstance(played_cards[0], list):
        for player_set in played_cards:
            player_cards = is_none(fillup_card_list(player_set))
            played_card_list.append(player_cards)
    else:
        played_card_list = None
    trumpf = trumpf_converter(game_type)

    if trumpf is None:
        return None

    return create_input(hand_card_list, table_card_list, played_card_list, trumpf)


def manuel_test_input_predict(model=None, hand_cards=None, table_cards=None, played_cards=None, game_type=None):
    hand_cards = is_none(hand_cards)
    table_cards = is_none(table_cards)
    played_cards = is_none(played_cards)
    game_type = is_none(game_type)

    g_model = get_model(model)

    input_matrix = create_test_matrix(hand_cards, table_cards, played_cards, game_type)

    if input_matrix is None:
        print("Your input is wrong. Please check all cards and the trumpf!")
    else:
        result = g_model.predict(np.asarray(input_matrix))
        max_value = np.argmax(result)
        card = Card.form_idx(max_value)
        print('Follow you see the list of predictions: \n {} \n'.format(result))
        print(
            'The model predict the card {} from {} with a probability of {} as the best one.'
            .format(card.number, card.color.name, result[0][max_value]))


# Set with model that model you want to test.
# Fill in all cards you know:
#   * hand_cards are all cards you actually hold in your hand  Input all cards you know your hand now.
#   * table_cards are all cards that lie now on the table
#   * played_cards are all cards that you know which have been played. Separate the played cards per player.
#  Attention: Set the game_type with one of those:
#   'DIAMONDS', 'HEARTS', 'SPADES', 'CLUBS', 'OBEABE', 'UNDEUFE'
if __name__ == '__main__':
    manuel_test_input_predict(model="./config/game_network_model_init__2017-11-28_224557.h5",
                              hand_cards=["H6", "H8", "H9", "HJ", "HA", "D9", "D10", "S8", "S10"],
                              played_cards=[],
                              game_type="HEARTS")
