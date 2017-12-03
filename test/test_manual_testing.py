import pytest
from elbotto.bots.training.card_parser import create_card
from elbotto.bots.training.manual_testing import is_none, get_model, fillup_card_list


@pytest.mark.parametrize("is_none_input, is_none_expected", [
    (None, []),
    ("Hello World!", "Hello World!")
])
def test_is_none_check(is_none_input, is_none_expected):
    assert is_none(is_none_input) == is_none_expected


def test_get_model_no_model():
    assert get_model(None) is None


@pytest.mark.parametrize("input_list, expected_list", [
    ([], []),
    (["DQ"], [create_card("DQ")]),
    (["DQ", "SA", "C8"], [create_card("DQ"), create_card("SA"), create_card("C8")]),
    (None, []),
    ("Test", None),
    (["Jass", "Tournament", 2017, True], [None, None, None, None])
])
def test_fillup_card_list_eval(input_list, expected_list):
    assert fillup_card_list(input_list) == expected_list
