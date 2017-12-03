import json
import pytest
from elbotto.bots.training.card_parser import create_card
from elbotto.bots.training.parser import check_path, check_file, print_trumpf, print_table
from elbotto.bots.training.parser import get_trumpf, get_remaining_hand_cards, complete_hand_cards_with_stiches


@pytest.mark.parametrize("input_path, expected", [
    ("/bla/", None),
    ("./test/parser_test/", "./test/parser_test/")
])
def test_check_path_eval_dir(input_path, expected):
    assert check_path(input_path) == expected


def test_check_file_invalid_inputs():
    assert check_file("./test/", "*.log") is None


def test_check_file_valid_file():
    input_string = ["./test/parser_test/testfile.txt"]
    result = check_file("./test/parser_test/", "testfile.txt")

    assert result == input_string


def test_get_trumpf_invalid():
    no_trumpf_string = json.loads(json.dumps(
        {"dealer": 2, "tricks": [{"cards": ["CK", "H7", "DA", "HQ"], "points": 19, "win": 3, "first": 1}]}))
    result = get_trumpf(no_trumpf_string)

    assert result is None


def test_get_trumpf_valid():
    trumpf_string = json.loads(json.dumps(
        {"trump": 4, "dealer": 2, "tricks": [{"cards": ["CK", "H7", "DA", "HQ"], "points": 19, "win": 3, "first": 1}]}))
    trumpf = get_trumpf(trumpf_string)

    assert trumpf.mode == "OBEABE"


def test_get_remaining_hand_cards_empty_hands():
    table = []
    end_hands = [{'hand': []}, {'hand': []}, {'hand': []}, {'hand': []}]
    amount_player = len(end_hands)
    table = get_remaining_hand_cards(end_hands, amount_player, table)

    assert table == [[], [], [], []]


def test_get_remaining_hand_cards_no_hand():
    table = []
    end_hands = [{'feet': []}, {'hand': []}, {'hand': []}, {'hand': []}]
    amount_player = len(end_hands)
    table = get_remaining_hand_cards(end_hands, amount_player, table)

    assert table == []


def test_get_remaining_hand_cards_handcards():
    table = []
    end_hands = [{'hand': ['DA', 'DK']},
                 {'hand': ['CA', 'CK']},
                 {'hand': ['HA', 'HK']},
                 {'hand': ['SA', 'SK']}]
    amount_player = len(end_hands)

    hands = ['DA', 'DK', 'CA', 'CK', 'HA', 'HK', 'SA', 'SK']
    target_table = [[], [], [], []]
    player = 0
    for c in range(len(hands)):
        card = create_card(hands[c])
        target_table[player].append(card)
        if c % 2 != 0:
            player += 1

    table = get_remaining_hand_cards(end_hands, amount_player, table)

    assert table == target_table


def test_complete_hand_cards_with_stiches_empty_table():
    stich_list = [{'cards': ['SJ', 'S10', 'S9', 'SA'], 'first': 0, 'points': 55, 'win': 0}]
    amount_players = 4
    table = []
    table = complete_hand_cards_with_stiches(stich_list, amount_players, table)

    assert table == 0


def test_complete_hand_cards_with_stiches_no_stich():
    stich_list = []
    amount_players = 4
    table = [[], [], [], []]
    table = complete_hand_cards_with_stiches(stich_list, amount_players, table)

    assert table == 0


def test_complete_hand_cards_with_stiches_empty_positions_on_table():
    stich_list = [{'cards': ['D10', 'S8', 'D7', 'D9'], 'first': 0},
                  {'cards': ['D6', 'SJ', 'CK', 'CQ'], 'first': 0},
                  {'cards': ['HJ', 'HQ', 'H9', 'HA'], 'first': 0},
                  {'cards': ['CJ', 'C10', 'S6', 'S9'], 'first': 0}]
    amount_players = 4
    table = [[], [], [], []]

    target_table = [[], [], [], []]
    stichs = ['D10', 'D9', 'D7', 'S8',
              'D6', 'CQ', 'CK', 'SJ',
              'HJ', 'HA', 'H9', 'HQ',
              'CJ', 'S9', 'S6', 'C10']
    for c in range(len(stichs)):
        player = c % 4
        card = create_card(stichs[c])
        target_table[player].append(card)

    table = complete_hand_cards_with_stiches(stich_list, amount_players, table)

    assert table == target_table


def test_complete_hand_cards_with_stiches_part_fully_hands():
    stich_list = [{'cards': ['D6', 'SJ', 'S6', 'S9'], 'first': 1},
                  {'cards': ['HJ', 'HQ', 'H9', 'HA'], 'first': 1},
                  {'cards': ['CJ', 'C10', 'CK', 'CQ'], 'first': 3}]
    amount_players = 4
    table = [[create_card("D10")], [create_card("S8")], [create_card("D7")], [create_card("D9")]]

    target_table = [[], [], [], []]
    stichs = ['D10', 'S8', 'D7', 'D9',
              'SJ', 'D6', 'S9', 'S6',
              'HQ', 'HJ', 'HA', 'H9',
              'CQ', 'CK', 'C10', 'CJ']
    for c in range(len(stichs)):
        player = c % 4
        card = create_card(stichs[c])
        target_table[player].append(card)

    table = complete_hand_cards_with_stiches(stich_list, amount_players, table)

    assert table == target_table


def test_print_trumpf_false_object():
    assert print_trumpf("TRUMPF") == 0


@pytest.mark.parametrize("input_value, expected_value", [
    ("Hello Test", 0),
    ([[], [], [], []], 0),
    ([], 0)
])
def test_print_table_eval(input_value, expected_value):
    assert print_table(input_value) == expected_value
