import glob
import json

from elbotto.bots.training import training_network as trainnet
from elbotto.bots.training.card_parser import Card_Parser
from elbotto.bots.training.trumpf_converter import Message, Trumpf_Color


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

                # Round complete with all hand cards for all players and trump
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

                        print("current Player: " + str(current_player))
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


if __name__ == '__main__':
    start_training()
