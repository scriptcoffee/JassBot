import numpy as np
from elbotto.bots.training.manual_testing import is_none, get_model, fillup_card_list
from elbotto.bots.training.trumpf_converter import trumpf_converter
from elbotto.bots.training.trumpf_training import create_input


def create_test_matrix(hand_cards, pushed_from_partner=False):
    if isinstance(pushed_from_partner, bool) and pushed_from_partner:
        shift = trumpf_converter(6)
    else:
        shift = trumpf_converter(0)

    hand_card_list = fillup_card_list(hand_cards)

    return create_input(hand_card_list, shift)


def manuel_test_input_predict(model=None, hand_cards=None, pushed=False):
    hand_cards = is_none(hand_cards)

    t_model = get_model(model)

    input_matrix = create_test_matrix(hand_cards, pushed)

    if input_matrix is None:
        print("Your input is wrong. Please check your cards!")
    else:
        result = t_model.predict(np.asarray(input_matrix))
        print(
            'The prediction is: '
            '\n hearts: {} \n diamonds: {} \n clubs: {} \n spades: {} \n OBEABE: {} \n UNDEUFE: {} \n SCHIEBE: {}'
            .format(result[0][0], result[0][1], result[0][2], result[0][3],
                    result[0][4], result[0][5], result[0][6]))


# Input your hand cards into the list and set 'pushed' to True if your partner moved the trump decision to you.
# Set with model that model you want to test.
if __name__ == '__main__':
    manuel_test_input_predict(model="./config/trumpf_network_model__2017-12-02_151805.h5",
                              hand_cards=["H6", "H8", "H9", "HJ", "HA", "D9", "D10", "S8", "S10"],
                              pushed=True)
