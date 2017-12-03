from elbotto.bots.training.manual_trumpf_testing import create_test_matrix


def test_create_test_matrix_pushed_from_partner():
    expected_input_matrix = [[1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
                              1]]

    hand_cards = ["H6", "H8", "H9", "HJ", "HA", "D9", "D10", "S8", "S10"]
    pushed_from_partner = True

    result = create_test_matrix(hand_cards, pushed_from_partner)

    assert (result == expected_input_matrix).all()


def test_create_test_matrix_not_pushed():
    expected_input_matrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
                              1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                              0]]

    hand_cards = ["C6", "C8", "C9", "CJ", "CA", "S9", "S10", "D8", "D10"]

    result = create_test_matrix(hand_cards)

    assert (result == expected_input_matrix).all()


def test_create_test_matrix_invalid_pushed_value():
    expected_input_matrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
                              1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                              0]]

    hand_cards = ["C6", "C8", "C9", "CJ", "CA", "S9", "S10", "D8", "D10"]
    pushed_from_partner = "geschoben"

    result = create_test_matrix(hand_cards, pushed_from_partner)

    assert (result == expected_input_matrix).all()


def test_create_test_matrix_eight_hand_cards():
    hand_cards = ["H6", "H8", "H9", "HJ", "CA", "S9", "S10", "D8"]

    result = create_test_matrix(hand_cards)

    assert result is None


def test_create_test_matrix_ten_hand_cards():
    hand_cards = ["C6", "C8", "H6", "H8", "H9", "HJ", "CA", "S9", "S10", "D8"]

    result = create_test_matrix(hand_cards)

    assert result is None


def test_create_test_matrix_no_hand_cards():
    hand_cards = []

    result = create_test_matrix(hand_cards)

    assert result is None
