import pytest
from elbotto.bots.training.card_parser import create_card
from elbotto.bots.training.game_parser import get_active_playing_cards, get_target_card, get_played_cards


@pytest.mark.parametrize("input_stich, expected_tuple", [
    ({'win': 1, 'points': 12, 'first': 1, 'cards': ['D7', 'DJ', 'S7', 'S10']},
     ([create_card("D7"), create_card("DJ"), create_card("S7")], create_card("S10"))),
    ({'win': 1, 'points': 12, 'first': 2, 'cards': ['D7', 'DJ', 'S7', 'S10']},
     ([], create_card("D7")))
])
def test_get_active_playing_cards_eval(input_stich, expected_tuple):
    amount_players = 4
    learning_player = 2

    assert get_active_playing_cards(input_stich, amount_players, learning_player) == expected_tuple


def test_get_target_card_from_last_player():
    expected_card = create_card("D7")

    input_stich = {'win': 1, 'points': 12, 'first': 1, 'cards': ['D7', 'DJ', 'S7', 'S10']}
    player_seat = 0

    assert get_target_card(player_seat, input_stich) == expected_card


@pytest.mark.parametrize("input_table, input_stich_number, expected_table", [
    ([[create_card("H9"), create_card("HJ"), create_card("S8"), create_card("SK"),
       create_card("C8"), create_card("C10"), create_card("H6"), create_card("DJ"), create_card("D10")],
      [create_card("DA"), create_card("DK"), create_card("CJ"), create_card("HA"),
       create_card("HQ"), create_card("D9"), create_card("HK"), create_card("S7"), create_card("S10")],
      [create_card("D8"), create_card("D6"), create_card("DQ"), create_card("D7"),
       create_card("C6"), create_card("SJ"), create_card("S9"), create_card("SQ"), create_card("H10")],
      [create_card("CQ"), create_card("CA"), create_card("C9"), create_card("S6"),
       create_card("CK"), create_card("H7"), create_card("C7"), create_card("SA"), create_card("H8")]],
     9,
     [[create_card("H9"), create_card("HJ"), create_card("S8"), create_card("SK"),
       create_card("C8"), create_card("C10"), create_card("H6"), create_card("DJ"), create_card("D10")],
      [create_card("DA"), create_card("DK"), create_card("CJ"), create_card("HA"),
       create_card("HQ"), create_card("D9"), create_card("HK"), create_card("S7"), create_card("S10")],
      [create_card("D8"), create_card("D6"), create_card("DQ"), create_card("D7"),
       create_card("C6"), create_card("SJ"), create_card("S9"), create_card("SQ"), create_card("H10")],
      [create_card("CQ"), create_card("CA"), create_card("C9"), create_card("S6"),
       create_card("CK"), create_card("H7"), create_card("C7"), create_card("SA"), create_card("H8")]]),

    ([[create_card("H9"), create_card("HJ"), create_card("S8"), create_card("SK"),
       create_card("C8"), create_card("C10"), create_card("H6"), create_card("DJ"), create_card("D10")],
      [create_card("DA"), create_card("DK"), create_card("CJ"), create_card("HA"),
       create_card("HQ"), create_card("D9"), create_card("HK"), create_card("S7"), create_card("S10")],
      [create_card("D8"), create_card("D6"), create_card("DQ"), create_card("D7"),
       create_card("C6"), create_card("SJ"), create_card("S9"), create_card("SQ"), create_card("H10")],
      [create_card("CQ"), create_card("CA"), create_card("C9"), create_card("S6"),
       create_card("CK"), create_card("H7"), create_card("C7"), create_card("SA"), create_card("H8")]],
     0, [[], [], [], []]),

    (([[create_card("H9"), create_card("HJ"), create_card("S8"), create_card("SK"),
       create_card("C8"), create_card("C10"), create_card("H6"), create_card("DJ"), create_card("D10")],
      [create_card("DA"), create_card("DK"), create_card("CJ"), create_card("HA"),
       create_card("HQ"), create_card("D9"), create_card("HK"), create_card("S7"), create_card("S10")],
      [create_card("D8"), create_card("D6"), create_card("DQ"), create_card("D7"),
       create_card("C6"), create_card("SJ"), create_card("S9"), create_card("SQ"), create_card("H10")],
      [create_card("CQ"), create_card("CA"), create_card("C9"), create_card("S6"),
       create_card("CK"), create_card("H7"), create_card("C7"), create_card("SA"), create_card("H8")]],
     4,
     [[create_card("H9"), create_card("HJ"), create_card("S8"), create_card("SK")],
      [create_card("DA"), create_card("DK"), create_card("CJ"), create_card("HA")],
      [create_card("D8"), create_card("D6"), create_card("DQ"), create_card("D7")],
      [create_card("CQ"), create_card("CA"), create_card("C9"), create_card("S6")]]))
])
def test_get_played_cards_eval(input_table, input_stich_number, expected_table):
    amount_players = 4

    assert get_played_cards(amount_players, input_stich_number, input_table) == expected_table
