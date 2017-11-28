from elbotto.bots.training.trumpf_training import create_input, create_target
from elbotto.bots.training.card_parser import create_card
from elbotto.bots.training.trumpf_converter import trumpf_converter


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


def test_create_input_incomplete_hand():
    hand_cards = [create_card("H9"),
                  create_card("HQ"),
                  create_card("C7"),
                  create_card("SA"),
                  create_card("S7"),
                  create_card("CJ"),
                  create_card("D7"),
                  create_card("CK")]
    shift = trumpf_converter(6)

    input_layer = create_input(hand_cards, shift)

    assert input_layer is None


def test_create_input_more_handcards():
    hand_cards = [create_card("H9"),
                  create_card("HQ"),
                  create_card("C7"),
                  create_card("SA"),
                  create_card("S7"),
                  create_card("CJ"),
                  create_card("CK"),
                  create_card("HK"),
                  create_card("D7"),
                  create_card("C6")]
    shift = trumpf_converter(6)

    input_layer = create_input(hand_cards, shift)

    assert input_layer is None


def test_create_input_double_cards():
    hand_cards = [create_card("H10"),
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

    assert input_layer is None


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


def test_create_target_chosen_shift():
    expectet_output_layer = [[0, 0, 0, 0, 0, 0, 1]]

    output_layer = create_target(trumpf_converter(6))

    assert (output_layer == expectet_output_layer).all()


def test_create_target_chosen_hearts():
    expectet_output_layer = [[1, 0, 0, 0, 0, 0, 0]]

    output_layer = create_target(trumpf_converter(1))

    assert (output_layer == expectet_output_layer).all()


def test_create_target_false_string():
    output_layer = create_target("clubs")

    assert output_layer is None


def test_create_target_chosen_diamonds_by_number():
    expectet_output_layer = [[0, 1, 0, 0, 0, 0, 0]]

    output_layer = create_target(0)

    assert (output_layer == expectet_output_layer).all()


def test_create_target_chosen_undeufe_by_name():
    expectet_output_layer = [[0, 0, 0, 0, 0, 1, 0]]

    output_layer = create_target("UNDEUFE")

    assert (output_layer == expectet_output_layer).all()