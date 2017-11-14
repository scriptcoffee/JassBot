from elbotto.card import Card as CardClass


class CardParser(CardClass):
    def __init__(self, number, color):
        super(CardParser, self).__init__(number, color)

    @staticmethod
    def create_card(card_symbol):
        color = card_symbol[0]
        if color == "H":
            color = "HEARTS"
        if color == "D":
            color = "DIAMONDS"
        if color == "C":
            color = "CLUBS"
        if color == "S":
            color = "SPADES"

        if card_symbol[1:] == "A":
            number = 14
        elif card_symbol[1:] == "K":
            number = 13
        elif card_symbol[1:] == "Q":
            number = 12
        elif card_symbol[1:] == "J":
            number = 11
        else:
            number = int(card_symbol[1:])

        return CardParser(number, color)
