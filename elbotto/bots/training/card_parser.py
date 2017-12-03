from elbotto.card import Card, CARD_OFFSET


class CardParser(Card):
    def __init__(self, number, color):
        super(CardParser, self).__init__(number, color)


def is_card_invalid(card_symbol):
    return card_symbol is None or isinstance(card_symbol, int) or len(card_symbol) <= 1


def create_card(card_symbol):
    if is_card_invalid(card_symbol):
        return None

    color_code = card_symbol[0]
    number_code = card_symbol[1:]

    if color_code in COLOR_DICT.keys():
        color = COLOR_DICT[color_code]
    else:
        return None

    if number_code in NUMBER_DICT:
        number = NUMBER_DICT[number_code]
    elif number_code.isdigit() and CARD_OFFSET <= int(number_code) < 11:
        number = int(number_code)
    else:
        return None

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
