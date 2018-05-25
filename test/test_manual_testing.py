import pytest
from elbotto.bots.training.card_parser import create_card
from elbotto.bots.training.manual_testing import is_none, get_model, fillup_card_list, create_test_matrix


@pytest.mark.parametrize("is_none_input, is_none_expected", [
    (None, []),
    ("Hello World!", "Hello World!")
])
def test_is_none_check(is_none_input, is_none_expected):
    assert is_none(is_none_input) == is_none_expected


def test_get_model_no_model():
    assert get_model(None) is None


@pytest.mark.parametrize("input_list, expected_list", [
    ([], []),
    (["DQ"], [create_card("DQ")]),
    (["DQ", "SA", "C8"], [create_card("DQ"), create_card("SA"), create_card("C8")]),
    (None, []),
    ("Test", None),
    (["Jass", "Tournament", 2017, True], [None, None, None, None])
])
def test_fillup_card_list_eval(input_list, expected_list):
    assert fillup_card_list(input_list) == expected_list


@pytest.mark.parametrize("hand_cards, pushed_from_partner, expected_input_matrix", [
    (["H6", "H8", "H9", "HJ", "HA", "D9", "D10", "S8", "S10"], True,
     [[1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1]]),
    (["C6", "C8", "C9", "CJ", "CA", "S9", "S10", "D8", "D10"], None,
     [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]]),
    (["C6", "C8", "C9", "CJ", "CA", "S9", "S10", "D8", "D10"], "geschoben",
     [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]])
])
def test_create_test_matrix_eval(hand_cards, pushed_from_partner, expected_input_matrix):
    if pushed_from_partner is None:
        result = create_test_matrix(hand_cards)
    else:
        result = create_test_matrix(hand_cards, pushed_from_partner)

    assert (result == expected_input_matrix).all()


@pytest.mark.parametrize("input_cards, expected_value", [
    (["H6", "H8", "H9", "HJ", "CA", "S9", "S10", "D8"], None),
    (["C6", "C8", "H6", "H8", "H9", "HJ", "CA", "S9", "S10", "D8"], None),
    ([], None)
])
def test_create_test_matrix_not_nine_hand_cards(input_cards, expected_value):
    assert create_test_matrix(input_cards) == expected_value