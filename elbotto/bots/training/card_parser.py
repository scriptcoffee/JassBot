from elbotto.card import Card as CardClass


class Card_Parser(CardClass):

    def __init__(self, number, color):
        super(Card_Parser, self).__init__(number, color)

    @staticmethod
    def create_card(input):
        color = input[0]
        if color == "H":
            color = "HEARTS"
        if color == "D":
            color = "DIAMONDS"
        if color == "C":
            color = "CLUBS"
        if color == "S":
            color = "SPADES"

        if input[1:] == "A":
            number = 14
        elif input[1:] == "K":
            number = 13
        elif input[1:] == "Q":
            number = 12
        elif input[1:] == "J":
            number = 11
        else:
            number = int(input[1:])

        return Card_Parser(number, color)