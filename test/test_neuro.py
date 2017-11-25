from elbotto.bots.neuro import PlayStrategy, TRUMPF_DICT
from elbotto.card import Card, Color


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

    ps = PlayStrategy(False)
    ps.INPUT_LAYER = 186

    input_layer = PlayStrategy.prepare_game_input(ps, game_type, hand_cards, table_cards, played_cards)

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

    ps = PlayStrategy(False)
    ps.INPUT_LAYER = 186

    input_layer = PlayStrategy.prepare_game_input(ps, game_type, hand_cards, table_cards, played_cards)

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

    ps = PlayStrategy(False)
    ps.INPUT_LAYER = 186

    input_layer = PlayStrategy.prepare_game_input(ps, game_type, hand_cards, table_cards, played_cards)

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

    game_type = TRUMPF_DICT[4]()
    hand_cards = []
    table_cards = []
    played_cards = []

    ps = PlayStrategy(False)
    ps.INPUT_LAYER = 186

    input_layer = PlayStrategy.prepare_game_input(ps, game_type, hand_cards, table_cards, played_cards)

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

    ps = PlayStrategy(False)
    ps.INPUT_LAYER = 186

    input_layer = PlayStrategy.prepare_game_input(
        ps,
        game_type=game_type,
        hand_cards=all_cards,
        table_cards=table_cards,
        played_cards=all_cards)

    assert (input_layer == expected_input_layer).all()


def get_all_cards():
    cards = []
    for color in Color:
        for i in range(6, 15):
            cards.append(Card.create(i, color.name))

    return cards