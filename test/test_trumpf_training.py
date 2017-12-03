import pytest
from elbotto.bots.training.card_parser import create_card
from elbotto.bots.training.trumpf_converter import trumpf_converter
from elbotto.bots.training.trumpf_training import create_input, create_target


def test_create_input_full_hand():
    expected_input_layer = [[0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                             0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
                             0]]

    hand_cards = [create_card("H9"),
                  create_card("DK"),
                  create_card("HQ"),
                  create_card("C7"),
                  create_card("SA"),
                  create_card("H10"),
                  create_card("S7"),
                  create_card("CJ"),
                  create_card("CK")]
    no_shift = trumpf_converter(2)

    input_layer = create_input(hand_cards, no_shift)

    assert (input_layer == expected_input_layer).all()


def test_create_input_hand_pushed():
    expected_input_layer = [[0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                             0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
                             1]]

    hand_cards = [create_card("H9"),
                  create_card("DK"),
                  create_card("HQ"),
                  create_card("C7"),
                  create_card("SA"),
                  create_card("H10"),
                  create_card("S7"),
                  create_card("CJ"),
                  create_card("CK")]
    shift = trumpf_converter(6)

    input_layer = create_input(hand_cards, shift)

    assert (input_layer == expected_input_layer).all()


@pytest.mark.parametrize("input_hand_cards, expected_value", [
    ([create_card("H9"), create_card("HQ"), create_card("C7"),
      create_card("SA"), create_card("S7"), create_card("CJ"),
      create_card("D7"), create_card("CK")],
     None),
    ([create_card("H9"), create_card("HQ"), create_card("C7"),
      create_card("SA"), create_card("S7"), create_card("CJ"),
      create_card("CK"), create_card("HK"), create_card("D7"), create_card("C6")],
     None),
    ([create_card("DK"), create_card("HQ"), create_card("C7"),
      create_card("SA"), create_card("S7"), create_card("CJ"),
      create_card("CK"), create_card("H10"),create_card("H10")],
     None)
])
def test_create_input_different_hand_cards(input_hand_cards, expected_value):
    assert create_input(input_hand_cards, trumpf_converter(6)) is expected_value


def test_create_input_trumpf_as_boolean():
    expected_input_layer = [[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                             0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1,
                             0]]

    hand_cards = [create_card("H9"),
                  create_card("HQ"),
                  create_card("C7"),
                  create_card("SA"),
                  create_card("S7"),
                  create_card("DA"),
                  create_card("SJ"),
                  create_card("CJ"),
                  create_card("CK")]

    input_layer = create_input(hand_cards, True)

    assert (input_layer == expected_input_layer).all()


def test_create_target_false_string():
    assert create_target("clubs") is None


@pytest.mark.parametrize("input_target, expected_target", [
    (trumpf_converter(6), [[0, 0, 0, 0, 0, 0, 1]]),
    (trumpf_converter(1), [[1, 0, 0, 0, 0, 0, 0]]),
    (0, [[0, 1, 0, 0, 0, 0, 0]]),
    ("UNDEUFE", [[0, 0, 0, 0, 0, 1, 0]])
])
def test_create_target_eval(input_target, expected_target):
    assert (create_target(input_target) == expected_target).all()
