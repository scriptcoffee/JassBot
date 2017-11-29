from elbotto.bots.neuro import prepare_game_input, TRUMPF_DICT, evaluate_trumpf_choise
from elbotto.card import Card, Color


def get_all_cards():
    cards = []
    for color in Color:
        for i in range(6, 15):
            cards.append(Card.create(i, color.name))

    return cards


def test_prepare_game_input_first_stich():
    expected_input_layer = \
    [[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0]]

    input_layer_size = 186
    game_type = TRUMPF_DICT[0]()
    hand_cards = [
        Card.create(6, "HEARTS"),
        Card.create(14, "CLUBS"),
        Card.create(7, "DIAMONDS"),
        Card.create(8, "HEARTS"),
        Card.create(13, "DIAMONDS"),
        Card.create(6, "SPADES"),
        Card.create(12, "SPADES"),
        Card.create(10, "DIAMONDS"),
        Card.create(7, "CLUBS")]
    table_cards = [{'number': 8, 'color': "CLUBS"}, {'number': 9, 'color': "SPADES"}]
    played_cards = []

    input_layer = prepare_game_input(input_layer_size, game_type, hand_cards, table_cards, played_cards)

    assert (input_layer == expected_input_layer).all()


def test_prepare_game_input_second_stich():
    expected_input_layer = \
    [[1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,
      0,  1,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  1,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  1,
      0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  1,  0,  0,  0]]

    input_layer_size = 186
    game_type = TRUMPF_DICT[2]()
    hand_cards = [
        Card.create(6, "HEARTS"),
        Card.create(14, "CLUBS"),
        Card.create(7, "DIAMONDS"),
        Card.create(8, "HEARTS"),
        Card.create(13, "DIAMONDS"),
        Card.create(6, "SPADES"),
        Card.create(12, "SPADES"),
        Card.create(7, "CLUBS")]

    table_cards = [
        {'number': 12, 'color': "DIAMONDS"},
        {'number': 7, 'color': "SPADES"},
        {'number': 9, 'color': "DIAMONDS"}]

    played_cards = [
        Card.create(10, "DIAMONDS"),
        Card.create(14, "DIAMONDS"),
        Card.create(8, "DIAMONDS"),
        Card.create(8, "CLUBS")]

    input_layer = prepare_game_input(input_layer_size, game_type, hand_cards, table_cards, played_cards)

    assert (input_layer == expected_input_layer).all()


def test_prepare_game_input_last_stich():
    expected_input_layer = \
    [[0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,
      1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
      0,  0,  0,  1,  0,  0]]

    input_layer_size = 186
    game_type = TRUMPF_DICT[3]()
    hand_cards = [Card.create(8, "HEARTS")]
    table_cards = []
    played_cards = [
        Card.create(6, "HEARTS"),
        Card.create(10, "HEARTS"),
        Card.create(7, "HEARTS"),
        Card.create(14, "HEARTS"),
        Card.create(11, "HEARTS"),
        Card.create(12, "HEARTS"),
        Card.create(13, "HEARTS"),
        Card.create(10, "DIAMONDS"),
        Card.create(6, "DIAMONDS"),
        Card.create(11, "DIAMONDS"),
        Card.create(12, "DIAMONDS"),
        Card.create(8, "DIAMONDS"),
        Card.create(7, "DIAMONDS"),
        Card.create(14, "DIAMONDS"),
        Card.create(9, "DIAMONDS"),
        Card.create(9, "CLUBS"),
        Card.create(10, "CLUBS"),
        Card.create(6, "CLUBS"),
        Card.create(8, "CLUBS"),
        Card.create(11, "CLUBS"),
        Card.create(14, "CLUBS"),
        Card.create(12, "CLUBS"),
        Card.create(13, "CLUBS"),
        Card.create(10, "SPADES"),
        Card.create(14, "SPADES"),
        Card.create(11, "SPADES"),
        Card.create(13, "SPADES"),
        Card.create(12, "SPADES"),
        Card.create(6, "SPADES"),
        Card.create(8, "SPADES"),
        Card.create(9, "SPADES"),
        Card.create(7, "SPADES")]

    input_layer = prepare_game_input(input_layer_size, game_type, hand_cards, table_cards, played_cards)

    assert (input_layer == expected_input_layer).all()


def test_prepare_game_input_none():
    expected_input_layer = \
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0]]

    input_layer_size = 186
    game_type = TRUMPF_DICT[4]()
    hand_cards = []
    table_cards = []
    played_cards = []

    input_layer = prepare_game_input(input_layer_size, game_type, hand_cards, table_cards, played_cards)

    assert (input_layer == expected_input_layer).all()


def test_prepare_game_input_all():
    expected_input_layer = \
    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      0, 0, 0, 0, 0, 1]]

    all_cards = get_all_cards()
    game_type = TRUMPF_DICT[5]()

    table_cards = [
        Card.create(6, "HEARTS").to_dict(),
        Card.create(6, "HEARTS").to_dict(),
        Card.create(6, "HEARTS").to_dict()]

    input_layer = prepare_game_input(
        input_layer_size=186,
        game_type=game_type,
        hand_cards=all_cards,
        table_cards=table_cards,
        played_cards=all_cards)

    assert (input_layer == expected_input_layer).all()


def test_trumpf_choise_evaluation_shift():
    hand_cards = [
        Card.create(13, "HEARTS"),
        Card.create(14, "HEARTS"),
        Card.create(8, "DIAMONDS"),
        Card.create(13, "DIAMONDS"),
        Card.create(8, "CLUBS"),
        Card.create(13, "CLUBS"),
        Card.create(9, "CLUBS"),
        Card.create(13, "SPADES"),
        Card.create(9, "SPADES")]

    assert evaluate_trumpf_choise(hand_cards, 6, 0) == 0


def test_trumpf_choise_evaluation_miss_shift():
    hand_cards = [
        Card.create(7, "HEARTS"),
        Card.create(14, "HEARTS"),
        Card.create(10, "DIAMONDS"),
        Card.create(11, "DIAMONDS"),
        Card.create(6, "CLUBS"),
        Card.create(7, "CLUBS"),
        Card.create(9, "CLUBS"),
        Card.create(13, "SPADES"),
        Card.create(12, "SPADES")]

    assert evaluate_trumpf_choise(hand_cards, 4, 0) == -30


def test_trumpf_choise_evaluation_color_trumpf():
    hand_cards = [
        Card.create(14, "HEARTS"),
        Card.create(11, "DIAMONDS"),
        Card.create(8, "DIAMONDS"),
        Card.create(13, "DIAMONDS"),
        Card.create(8, "CLUBS"),
        Card.create(6, "CLUBS"),
        Card.create(9, "CLUBS"),
        Card.create(14, "SPADES"),
        Card.create(9, "SPADES")]

    assert evaluate_trumpf_choise(hand_cards, 1, 0) == 0


def test_trumpf_choise_evaluation_undeufe():
    hand_cards = [
        Card.create(6, "HEARTS"),
        Card.create(7, "HEARTS"),
        Card.create(7, "DIAMONDS"),
        Card.create(8, "DIAMONDS"),
        Card.create(14, "CLUBS"),
        Card.create(6, "CLUBS"),
        Card.create(12, "CLUBS"),
        Card.create(7, "SPADES"),
        Card.create(6, "SPADES")]

    assert evaluate_trumpf_choise(hand_cards, 5, 0) == 0


def test_trumpf_choise_evaluation_wrong_trumpf():
    hand_cards = [
        Card.create(11, "HEARTS"),
        Card.create(14, "HEARTS"),
        Card.create(9, "HEARTS"),
        Card.create(13, "HEARTS"),
        Card.create(8, "CLUBS"),
        Card.create(13, "CLUBS"),
        Card.create(9, "CLUBS"),
        Card.create(14, "SPADES"),
        Card.create(9, "SPADES")]

    assert evaluate_trumpf_choise(hand_cards, 1, 0) == -59


def test_trumpf_choise_evaluation_shifted():
    hand_cards = [
        Card.create(7, "HEARTS"),
        Card.create(14, "HEARTS"),
        Card.create(10, "DIAMONDS"),
        Card.create(11, "DIAMONDS"),
        Card.create(6, "CLUBS"),
        Card.create(7, "CLUBS"),
        Card.create(9, "CLUBS"),
        Card.create(13, "SPADES"),
        Card.create(12, "SPADES")]

    assert evaluate_trumpf_choise(hand_cards, 1, 1) == 0


def test_trumpf_choise_evaluation_shifted_wrong_trumpf():
    hand_cards = [
        Card.create(7, "HEARTS"),
        Card.create(14, "HEARTS"),
        Card.create(10, "DIAMONDS"),
        Card.create(11, "DIAMONDS"),
        Card.create(6, "CLUBS"),
        Card.create(7, "CLUBS"),
        Card.create(9, "CLUBS"),
        Card.create(13, "SPADES"),
        Card.create(12, "SPADES")]

    assert evaluate_trumpf_choise(hand_cards, 3, 1) == -20


def test_trumpf_choise_evaluation_shifted_wrong_obenabe():
    hand_cards = [
        Card.create(6, "HEARTS"),
        Card.create(7, "HEARTS"),
        Card.create(9, "HEARTS"),
        Card.create(13, "DIAMONDS"),
        Card.create(12, "DIAMONDS"),
        Card.create(7, "CLUBS"),
        Card.create(14, "CLUBS"),
        Card.create(10, "SPADES"),
        Card.create(11, "SPADES")]

    assert evaluate_trumpf_choise(hand_cards, 4, 1) == -33
