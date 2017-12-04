import pytest
from elbotto.card import Card
from elbotto.bots.training.card_parser import create_card, is_card_invalid


@pytest.mark.parametrize("test_input, expected_boolean", [
    ("", True),
    (11, True),
    ("H", True),
    (None, True),
    ("OI", False)
])
def test_is_card_invalid_check(test_input, expected_boolean):
    assert is_card_invalid(test_input) is expected_boolean


@pytest.mark.parametrize("test_input_card, expected", [
    ("DA", Card.create(14, "DIAMONDS")),
    ("C6", Card.create(6, "CLUBS")),
    ("KK", None),
    ("HP", None),
    ("H5", None),
    ("S11", None),
    ("ha", None)
])
def test_create_card_eval(test_input_card, expected):
    assert create_card(test_input_card) == expected
