from elbotto.bots.training.game_training import create_input, create_target
from elbotto.bots.training.card_parser import create_card
from elbotto.bots.training.trumpf_converter import trumpf_converter


def test_create_input_no_played_cards():
    expected_input_layer = [[1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                             0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 1, 0, 0, 0, 0]]

    hand_cards = [create_card("C10"),
                  create_card("H6"),
                  create_card("CJ"),
                  create_card("H9"),
                  create_card("HJ"),
                  create_card("S8"),
                  create_card("HA"),
                  create_card("SK"),
                  create_card("DJ")]

    table_cards = [create_card("D10"), create_card("DK"), create_card("DA")]

    played_cards = []
    game_type_diamonds = trumpf_converter(0)

    input_layer = create_input(hand_cards, table_cards, played_cards, game_type_diamonds)

    assert (input_layer == expected_input_layer).all()


def test_create_input_valid_cards():
    expected_input_layer = [[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,

                             1, 0, 0, 0, 0, 0]]

    hand_cards = [create_card("H9"),
                  create_card("HJ"),
                  create_card("S8"),
                  create_card("SK")]

    table_cards = [create_card("C8")]

    played_cards = [[create_card("C10"),
                     create_card("H6"),
                     create_card("DJ"),
                     create_card("D10"),
                     create_card("DK")],
                    [create_card("DA"),
                     create_card("CJ"),
                     create_card("HA"),
                     create_card("HQ"),
                     create_card("D9")],
                    [create_card("D8"),
                     create_card("D6"),
                     create_card("DQ"),
                     create_card("D7"),
                     create_card("C6")],
                    [create_card("CQ"),
                     create_card("CA"),
                     create_card("C9"),
                     create_card("S6"),
                     create_card("CK")]]
    game_type_hearts = trumpf_converter(1)

    input_layer = create_input(hand_cards, table_cards, played_cards, game_type_hearts)

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

                             1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0,

                             0, 0, 0, 1, 0, 0]]

    hand_cards = [create_card("H9"),
                  create_card("HJ"),
                  create_card("S8")]

    table_cards = []

    played_cards = [[create_card("C10"),
                     create_card("H6"),
                     create_card("DJ"),
                     create_card("D10"),
                     create_card("DK"),
                     create_card("DA")],
                    [create_card("CJ"),
                     create_card("HA"),
                     create_card("HQ"),
                     create_card("D9"),
                     create_card("D8"),
                     create_card("D6")],
                    [create_card("DQ"),
                     create_card("D7"),
                     create_card("C6"),
                     create_card("CQ"),
                     create_card("CA"),
                     create_card("C9")],
                    [create_card("S6"),
                     create_card("CK"),
                     create_card("C8"),
                     create_card("H7"),
                     create_card("SK"),
                     create_card("SJ")]]
    game_type_spades = trumpf_converter(2)

    input_layer = create_input(hand_cards, table_cards, played_cards, game_type_spades)

    assert (input_layer == expected_input_layer).all()


def test_create_input_no_cards_per_player_played():
    expected_input_layer = [[1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                             0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 1, 0, 0, 0]]

    hand_cards = [create_card("C10"),
                  create_card("H6"),
                  create_card("CJ"),
                  create_card("H9"),
                  create_card("HJ"),
                  create_card("S8"),
                  create_card("HA"),
                  create_card("SK"),
                  create_card("DJ")]

    table_cards = [create_card("D10"), create_card("DK"), create_card("DA")]

    played_cards = [[], [], [], []]
    game_type_clubs = trumpf_converter(3)

    input_layer = create_input(hand_cards, table_cards, played_cards, game_type_clubs)

    assert (input_layer == expected_input_layer).all()


def test_create_input_no_hand_cards():
    hand_cards = []

    table_cards = [create_card("H9"),
                   create_card("HJ"),
                   create_card("S8")]

    played_cards = [[create_card("C10"),
                     create_card("H6"),
                     create_card("DJ"),
                     create_card("D10"),
                     create_card("DK"),
                     create_card("DA")],
                    [create_card("CJ"),
                     create_card("HA"),
                     create_card("HQ"),
                     create_card("D9"),
                     create_card("D8"),
                     create_card("D6")],
                    [create_card("DQ"),
                     create_card("D7"),
                     create_card("C6"),
                     create_card("CQ"),
                     create_card("CA"),
                     create_card("C9")],
                    [create_card("S6"),
                     create_card("CK"),
                     create_card("C8"),
                     create_card("H7"),
                     create_card("SK"),
                     create_card("SJ")]]
    game_type_obeabe = trumpf_converter(4)

    input_layer = create_input(hand_cards, table_cards, played_cards, game_type_obeabe)

    assert input_layer is None


def test_create_input_pure_list_of_cards_for_played_cards():
    expected_input_layer = [[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

                             1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,

                             0, 0, 0, 0, 0, 1, ]]

    hand_cards = [create_card("H9"),
                  create_card("HJ"),
                  create_card("S8"),
                  create_card("SK")]

    table_cards = [create_card("C8")]

    played_cards = [create_card("C10"),
                    create_card("H6"),
                    create_card("DJ"),
                    create_card("D10"),
                    create_card("DK"),
                    create_card("DA"),
                    create_card("CJ"),
                    create_card("HA"),
                    create_card("HQ"),
                    create_card("D9"),
                    create_card("D8"),
                    create_card("D6"),
                    create_card("DQ"),
                    create_card("D7"),
                    create_card("C6"),
                    create_card("CQ"),
                    create_card("CA"),
                    create_card("C9"),
                    create_card("S6"),
                    create_card("CK")]
    game_type_undeufe = trumpf_converter(5)

    input_layer = create_input(hand_cards, table_cards, played_cards, game_type_undeufe)

    assert (input_layer == expected_input_layer).all()


def test_create_target_true_validation_list():
    output_layer = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    target_list = create_target(create_card("CJ"))

    assert (target_list == output_layer).all()


def test_create_target_no_card():
    target_list = create_target("")

    assert target_list is None


def test_create_target_string():
    target_list = create_target("CJ")

    assert target_list is None


def test_create_target_int():
    target_list = create_target(14)

    assert target_list is None
