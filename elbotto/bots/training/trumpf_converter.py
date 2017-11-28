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
    def __init__(self, mode, color="", code=""):
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


def set_schiebe():
    return TrumpfCard("SCHIEBE")


TRUMPF_DICT = {0: set_diamonds,
               'DIAMONDS': set_diamonds,
               1: set_hearts,
               'HEARTS': set_hearts,
               2: set_spades,
               'SPADES': set_spades,
               3: set_clubs,
               'CLUBS': set_clubs,
               4: set_obeabe,
               'OBEABE': set_obeabe,
               5: set_undeufe,
               'UNDEUFE': set_undeufe,
               6: set_schiebe,
               'SCHIEBE': set_schiebe
               }


def trumpf_converter(trumpf_code):
    if trumpf_code in TRUMPF_DICT.keys():
        trumpf_card = TRUMPF_DICT[trumpf_code]()
    else:
        return None
    return trumpf_card
