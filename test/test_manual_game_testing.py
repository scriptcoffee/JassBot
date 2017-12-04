import pytest
from elbotto.bots.training.manual_game_testing import create_test_matrix


def test_create_test_matrix_valid_input():
    expected_input_layer = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,

                             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 1, 0, 0, 0, 0]]

    hand_cards = ["CA", "D6", "SJ"]
    table_cards = ["H6"]
    game_type = "DIAMONDS"

    result = create_test_matrix(hand_cards, table_cards, game_type)

    assert (result == expected_input_layer).all()


@pytest.mark.parametrize("input_hand, input_table, input_game_type, expected", [
    (["CA", "D6", "SJ"], ["H6"], "acorn", None),
    (["A", "D", "J"], ["H6"], "HEARTS", None)
])
def test_create_test_matrix_invalid_input(input_hand, input_table, input_game_type, expected):
    result = create_test_matrix(input_hand, input_table, input_game_type)
    assert result == expected


def test_create_test_matrix_empty_table():
    expected_input_layer = [[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 1]]

    hand_cards = ["HA", "H6", "CJ"]
    table_cards = []
    game_type = "UNDEUFE"

    result = create_test_matrix(hand_cards, table_cards, game_type)

    assert (result == expected_input_layer).all()
