from enum import Enum


class Trumpf(Enum):
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3


class Message(object):

    def __init__(self, card_type, trumpf_code):
        self.trumpf_code = trumpf_code
        self.mode = ""
        self.trumpf_color = card_type

    def trumpf_parser(self):
        if self.trumpf_code < 4:
            self.mode = "TRUMPF"
            if self.trumpf_code == 0:
                self.trumpf_color.name = Trumpf.DIAMONDS.name
                self.trumpf_color.value = Trumpf.DIAMONDS.value
            if self.trumpf_code == 1:
                self.trumpf_color.name = Trumpf.HEARTS.name
                self.trumpf_color.value = Trumpf.HEARTS.value
            if self.trumpf_code == 2:
                self.trumpf_color.name = Trumpf.SPADES.name
                self.trumpf_color.value = Trumpf.SPADES.value
            if self.trumpf_code == 3:
                self.trumpf_color.name = Trumpf.CLUBS.name
                self.trumpf_color.value = Trumpf.CLUBS.value
        if self.trumpf_code == 4:
            self.mode = "OBEABE"
        if self.trumpf_code == 5:
            self.mode = "UNDEUFE"
        return self


class Trumpf_Color(object):

    def __init__(self, color="", code=0):
        self.name = color
        self.value = code