import json
import pytest
from elbotto.bots.training.card_parser import create_card
from elbotto.bots.training.parser_helper import check_path, check_file, print_trumpf, print_table
from elbotto.bots.training.parser_helper import get_trumpf, get_remaining_hand_cards, complete_hand_cards_with_stiches


@pytest.mark.parametrize("input_path, expected", [
    ("/bla/", None),
    ("./test/parser_test/", "./test/parser_test/")
])
def test_check_path_eval_dir(input_path, expected):
    assert check_path(input_path) == expected


@pytest.mark.parametrize("input_path, input_file, expected_value", [
    ("./test/", "*.log", None),
    ("./test/parser_test/", "testfile.txt", ["./test/parser_test/testfile.txt"])
])
def test_check_file_eval(input_path, input_file, expected_value):
    assert check_file(input_path, input_file) == expected_value


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


@pytest.mark.parametrize("input_hands, input_table, expected_table", [
    ([{'hand': []}, {'hand': []}, {'hand': []}, {'hand': []}], [], [[], [], [], []]),
    ([{'feet': []}, {'hand': []}, {'hand': []}, {'hand': []}], [], [])
])
def test_get_remaining_hand_cards_eval(input_hands, input_table, expected_table):
    amount_player = len(input_hands)
    assert get_remaining_hand_cards(input_hands, amount_player, input_table) == expected_table


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


@pytest.mark.parametrize("input_stich, amount_players, input_table, expected_value", [
    ([{'cards': ['SJ', 'S10', 'S9', 'SA'], 'first': 0, 'points': 55, 'win': 0}], 4, [], 0),
    ([], 4, [[], [], [], []], 0)
])
def test_complete_hand_cards_with_stiches_invalid_eval(input_stich, amount_players, input_table, expected_value):
    assert complete_hand_cards_with_stiches(input_stich, amount_players, input_table) == expected_value


@pytest.mark.parametrize("input_stich_list, input_table, expected_stich", [
    ([{'cards': ['D10', 'S8', 'D7', 'D9'], 'first': 0}, {'cards': ['D6', 'SJ', 'CK', 'CQ'], 'first': 0},
      {'cards': ['HJ', 'HQ', 'H9', 'HA'], 'first': 0}, {'cards': ['CJ', 'C10', 'S6', 'S9'], 'first': 0}],
     [[], [], [], []],
     ['D10', 'D9', 'D7', 'S8', 'D6', 'CQ', 'CK', 'SJ', 'HJ', 'HA', 'H9', 'HQ', 'CJ', 'S9', 'S6', 'C10']),
    ([{'cards': ['D6', 'SJ', 'S6', 'S9'], 'first': 1}, {'cards': ['HJ', 'HQ', 'H9', 'HA'], 'first': 1},
      {'cards': ['CJ', 'C10', 'CK', 'CQ'], 'first': 3}],
     [[create_card("D10")], [create_card("S8")], [create_card("D7")], [create_card("D9")]],
     ['D10', 'S8', 'D7', 'D9', 'SJ', 'D6', 'S9', 'S6', 'HQ', 'HJ', 'HA', 'H9', 'CQ', 'CK', 'C10', 'CJ']),
])
def test_complete_hand_cards_with_stiches_valid_eval(input_stich_list, input_table, expected_stich):
    amount_players = 4

    target_table = [[], [], [], []]
    for c in range(len(expected_stich)):
        player = c % 4
        card = create_card(expected_stich[c])
        target_table[player].append(card)

    table = complete_hand_cards_with_stiches(input_stich_list, amount_players, input_table)

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
