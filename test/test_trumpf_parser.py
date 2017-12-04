import pytest
from elbotto.bots.training.card_parser import create_card
from elbotto.bots.training.trumpf_converter import trumpf_converter
from elbotto.bots.training.trumpf_parser import count_game_type, collect_trumpf_per_game


@pytest.mark.parametrize("input_game_type, input_trumpf_tuples, expected_tuples", [
    (trumpf_converter("CLUBS"), {}, {"CLUBS": 1}),
    (trumpf_converter("SPADES"), {"SPADES": 15}, {"SPADES": 16}),
    (trumpf_converter("OBEABE"), {"SPADES": 16}, {"SPADES": 16, "OBEABE": 1}),
])
def test_count_game_type_eval(input_game_type, input_trumpf_tuples, expected_tuples):
    assert count_game_type(input_game_type, input_trumpf_tuples) == expected_tuples


def test_collect_trumpf_per_game_with_tss():
    expected_hand_list = [[create_card("H9"), create_card("DK"), create_card("HQ"),
                           create_card("C7"), create_card("SA"), create_card("H10"),
                           create_card("S7"), create_card("CJ")], [create_card("SJ")]]
    expected_trumpf_list = [trumpf_converter("HEARTS"), trumpf_converter(2)]
    expected_trumpf_tuples = {'SPADES': 13, 'HEARTS': 6, 'OBEABE': 7}
    expected_tss = 16

    hand_list = [[create_card("H9"), create_card("DK"), create_card("HQ"),
                  create_card("C7"), create_card("SA"), create_card("H10"),
                  create_card("S7"), create_card("CJ")]]
    rounds = {'rounds': [{'dealer': 0, 'tss': 1,
                          'tricks': [{'points': 24, 'first': 3, 'cards': ['S8', 'S7', 'SJ', 'HA'], 'win': 3}],
                          'trump': 2, 'player': [{'hand': []}, {'hand': []}, {'hand': []}, {'hand': []}]}]}
    trumpf_list = [trumpf_converter("HEARTS")]
    trumpf_tumples = {'SPADES': 12, 'HEARTS': 6, 'OBEABE': 7}
    tss = 15

    result = collect_trumpf_per_game(hand_list, rounds, trumpf_list, trumpf_tumples, tss)

    assert result[0] == expected_hand_list
    assert result[1][0].trumpf_color.name == expected_trumpf_list[0].trumpf_color.name
    assert result[1][1].trumpf_color.name == expected_trumpf_list[1].trumpf_color.name
    assert result[2] == expected_trumpf_tuples
    assert result[3] == expected_tss
