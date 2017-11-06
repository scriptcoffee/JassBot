import json
import glob

from enum import Enum
from elbotto.card import Card as CardClass

from elbotto.bots import training_network as trainnet


def start_training():
    # create an instance of the model to want to train
    network = trainnet.Training("Supervised_Network")
    # Import and validate all dates
    files = glob.glob('C:\Programming\BA\ExterneLogfiles\sl\Logs\*.txt')
    for file_path in files:
        print(file_path)

        lines = open(file_path).readlines()

        for line in lines:

            game = line[43:]
            rounds = json.loads(game)

            #print(rounds)
            print("Game: " + str(line))

            amount_rounds = len(rounds['rounds'])
            amount_players = len(rounds['rounds'][0]['player'])
            print(amount_players)

            for i in range(amount_rounds):

                table = []

                print(str(i) + ". Round: " + str(rounds['rounds'][i]))

                if rounds['rounds'][i] is None:
                    # print("Type None isn't valid.")
                    break

                if 'trump' not in rounds['rounds'][i]:
                    # print("Round hasn't a trump, so we skip it")
                    break

                for player in range(amount_players):
                    if 'hand' not in rounds['rounds'][i]['player'][player]:
                        # print("Round has no hands, so we skip it")
                        break
                    table.insert(player, rounds['rounds'][i]['player'][player]['hand'])

                trumpf = rounds['rounds'][i]['trump']
                card_type = Trumpf_Color()
                trumpf_message = Message(card_type, int(trumpf))
                game_type = trumpf_message.trumpf_parser()
                amount_stich = len(rounds['rounds'][i]['tricks'])
                print(amount_stich)
                for stich in range(amount_stich):
                    current_player = int(rounds['rounds'][i]['tricks'][stich]['first'])
                    for player_seat in range(amount_players):
                        played_card = rounds['rounds'][i]['tricks'][stich]['cards'][player_seat]
                        card = Card_Parser.create_card(played_card)
                        table[current_player].append(card)
                        current_player = (current_player + 1) % amount_players

                # Round completet with all hand cards for all players and trump
                print("trumpf: " + str(game_type.mode))
                print("player0: " + str(table[0]))
                print("player1: " + str(table[1]))
                print("player2: " + str(table[2]))
                print("player3: " + str(table[3]))

                for learning_player in range(amount_players):

                    print("learning player: " + str(learning_player))

                    table_list = []
                    hand_list = []
                    trumpf_list = []
                    target_list = []

                    for stich in range(amount_stich):
                        cards_on_table = []
                        hand = table[learning_player][stich:amount_stich]
                        print("Stich: " + str(rounds['rounds'][i]['tricks'][stich]))
                        current_player = int(rounds['rounds'][i]['tricks'][stich]['first'])

                        print("aktueller Spieler: " + str(current_player))
                        player_seat = 0

                        while current_player != learning_player:
                            played_card = rounds['rounds'][i]['tricks'][stich]['cards'][player_seat]
                            card = Card_Parser.create_card(played_card)

                            cards_on_table.insert(player_seat, card)

                            current_player = (current_player + (amount_players - 1)) % amount_players
                            player_seat += 1

                        target = rounds['rounds'][i]['tricks'][stich]['cards'][learning_player]
                        print("Target from learning player: " + str(target))
                        target_card = Card_Parser.create_card(target)

                        print("Cards on table: " + str(cards_on_table))
                        table_list.append(cards_on_table)
                        print("Handcards: " + str(hand))
                        hand_list.append(hand)
                        print("Trumpf: " + str(game_type.mode))
                        trumpf_list.append(game_type)
                        print("Target: " + str(target_card))
                        target_list.append(target_card)

                        # call BotNetwork with hand, cards from table, trumpf and the list of all targets
                    network.train_the_model(hand_list, table_list, trumpf_list, target_list)


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




class Trumpf_Color(object):

    def __init__(self, color="", code=0):
        self.name = color
        self.value = code

if __name__ == '__main__':
    start_training()
