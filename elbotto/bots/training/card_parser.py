from elbotto.card import Card as CardClass


class CardParser(CardClass):
    def __init__(self, number, color):
        super(CardParser, self).__init__(number, color)


def create_card(card_symbol):
    color = colorDict[card_symbol[0]]
    if card_symbol[1:] in numberDict:
        number = numberDict[card_symbol[1:]]
    else:
        number = int(card_symbol[1:])
    return CardParser(number, color)


colorDict = {"H": "HEARTS",
             "D": "DIAMONDS",
             "C": "CLUBS",
             "S": "SPADES"
             }

numberDict = {"A": 14,
              "K": 13,
              "Q": 12,
              "J": 11
              }
