from elbotto.bots.training.card_parser import create_card
from elbotto.card import Card


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
    card = create_card("hd")

    assert card is None


def test_create_card_empty_object():
    card = create_card("")

    assert card is None


def test_create_card_int():
    card = create_card(15)

    assert card is None


def test_create_card_one_symbole():
    card = create_card("H")

    assert card is None


def test_create_card_none():
    card = create_card(None)

    assert card is None
