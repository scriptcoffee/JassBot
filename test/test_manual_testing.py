from elbotto.bots.training.manual_testing import is_none, get_model, fillup_card_list
from elbotto.bots.training.card_parser import create_card


def test_is_none_none():
    assert is_none(None) == []


def test_is_none_not_none():
    test_string = "Hello World!"

    assert is_none(test_string) == test_string


def test_get_model_no_model():
    assert get_model(None) is None


def test_fillup_card_list_empty_list():
    test_list = []
    result = fillup_card_list(test_list)

    assert result == test_list


def test_fillup_card_list_one_card():
    expected_card = [create_card("DQ")]

    card_list = ["DQ"]
    result = fillup_card_list(card_list)

    assert result == expected_card


def test_fillup_card_list_three_cards():
    expected_card = [create_card("DQ"), create_card("SA"), create_card("C8")]

    card_list = ["DQ", "SA", "C8"]
    result = fillup_card_list(card_list)

    assert result == expected_card


def test_fillup_card_list_none():
    result = fillup_card_list(None)

    assert result == []


def test_fillup_card_list_no_list():
    result = fillup_card_list("Test")

    assert result is None


def test_fillup_card_list_invalid_list():
    expected_list = [None, None, None, None]

    invalid_list = ["Jass", "Tournament", 2017, True]
    result = fillup_card_list(invalid_list)

    assert result == expected_list
