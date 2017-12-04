import pytest
from elbotto.bots.training.card_parser import create_card
from elbotto.bots.training.trumpf_converter import trumpf_converter
from elbotto.bots.training.game_training import create_input, create_target


def test_create_input_valid_cards():
    expected_input_layer = [[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             1, 0, 0, 0, 0, 0]]

    hand_cards = [create_card("H9"), create_card("HJ"), create_card("S8"), create_card("SK")]
    table_cards = [create_card("C8")]
    game_type_hearts = trumpf_converter(1)

    input_layer = create_input(hand_cards, table_cards, game_type_hearts)

    assert (input_layer == expected_input_layer).all()


def test_create_input_no_card_on_table():
    expected_input_layer = [[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 1, 0, 0]]

    hand_cards = [create_card("H9"), create_card("HJ"), create_card("S8")]
    table_cards = []
    game_type_spades = trumpf_converter(2)

    input_layer = create_input(hand_cards, table_cards, game_type_spades)

    assert (input_layer == expected_input_layer).all()


def test_create_input_no_hand_cards():
    hand_cards = []
    table_cards = [create_card("H9"), create_card("HJ"), create_card("S8")]
    game_type_obeabe = trumpf_converter(4)

    input_layer = create_input(hand_cards, table_cards, game_type_obeabe)

    assert input_layer is None


def test_create_target_true_validation_list():
    output_layer = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    target_list = create_target(create_card("CJ"))

    assert (target_list == output_layer).all()


@pytest.mark.parametrize("test_input, expected", [
    ("", None),
    ("CJ", None),
    (14, None)
])
def test_create_target_primitive_input(test_input, expected):
    assert create_target(test_input) == expected
