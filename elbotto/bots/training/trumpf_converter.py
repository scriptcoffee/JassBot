from enum import Enum


class Trumpf(Enum):
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3


class TrumpfColor:
    def __init__(self, color, code):
        self.name = color
        self.value = code


class TrumpfCard:
    def __init__(self, mode, color="", code=0):
        self.mode = mode
        self.trumpf_color = TrumpfColor(color, code)


def set_diamonds():
    return TrumpfCard("TRUMPF", Trumpf.DIAMONDS.name, Trumpf.DIAMONDS.value)


def set_hearts():
    return TrumpfCard("TRUMPF", Trumpf.HEARTS.name, Trumpf.HEARTS.value)


def set_spades():
    return TrumpfCard("TRUMPF", Trumpf.SPADES.name, Trumpf.SPADES.value)


def set_clubs():
    return TrumpfCard("TRUMPF", Trumpf.CLUBS.name, Trumpf.CLUBS.value)


def set_obeabe():
    return TrumpfCard("OBEABE")


def set_undeufe():
    return TrumpfCard("UNDEUFE")


TRUMPF_DICT = {0: set_diamonds,
               1: set_hearts,
               2: set_spades,
               3: set_clubs,
               4: set_obeabe,
               5: set_undeufe
               }


def trumpf_converter(trumpf_code):
    trumpf_card = TRUMPF_DICT[trumpf_code]()
    return trumpf_card
