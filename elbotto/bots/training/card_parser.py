from elbotto.card import Card as CardClass


class CardParser(CardClass):
    def __init__(self, number, color):
        super(CardParser, self).__init__(number, color)


def create_card(card_symbol):
    color = COLOR_DICT[card_symbol[0]]
    if card_symbol[1:] in NUMBER_DICT:
        number = NUMBER_DICT[card_symbol[1:]]
    else:
        number = int(card_symbol[1:])
    return CardParser(number, color)


COLOR_DICT = {"H": "HEARTS",
             "D": "DIAMONDS",
             "C": "CLUBS",
             "S": "SPADES"
              }

NUMBER_DICT = {"A": 14,
              "K": 13,
              "Q": 12,
              "J": 11
               }
