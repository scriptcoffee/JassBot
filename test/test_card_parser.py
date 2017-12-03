from elbotto.bots.training.card_parser import create_card, is_card_invalid
from elbotto.card import Card


def test_is_card_invalid_empty_object():
    assert is_card_invalid("") is True


def test_is_card_invalid_int():
    assert is_card_invalid(11) is True


def test_is_card_invalid_one_symbole():
    assert is_card_invalid("H") is True


def test_is_card_invalid_none():
    assert is_card_invalid(None) is True


def test_is_card_invalid_valid_card():
    assert is_card_invalid("OI") is False


def test_create_card_valid_head_card():
    card_log_format = create_card("DA")
    card_server_format = Card.create(14, "DIAMONDS")

    assert card_log_format == card_server_format


def test_create_card_valid_number_card():
    card_log_format = create_card("C6")
    card_server_format = Card.create(6, "CLUBS")

    assert card_log_format == card_server_format


def test_create_card_invalid_color():
    card = create_card("KK")

    assert card is None


def test_create_card_no_number():
    card = create_card("HP")

    assert card is None


def test_create_card_low_number():
    card = create_card("H5")

    assert card is None


def test_create_card_high_number():
    card = create_card("S11")

    assert card is None


def test_create_card_lower_case():
    card = create_card("ha")

    assert card is None
