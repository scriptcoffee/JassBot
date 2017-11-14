import glob
import json

from datetime import datetime
from keras import backend as k

from elbotto.bots.training import game_training as trainnet
from elbotto.bots.training.card_parser import CardParser
from elbotto.bots.training.trumpf_converter import Message, TrumpfColor


def start_training():
    # create an instance of the model to want to train
    network = trainnet.GameTraining("Supervised_Gamenetwork")
    # Import and validate all dates
    files = glob.glob('./data/*.txt')
    file_number = 0
    for file_path in files:
        print(file_path)

        lines = open(file_path).readlines()

        for line in lines:
            game = line[43:]
            rounds = json.loads(game)

            print("Game: " + str(line))

            amount_rounds = len(rounds['rounds'])
            amount_players = len(rounds['rounds'][0]['player'])

            for round in range(amount_rounds):

                table = []

                print(str(round) + ". Round: " + str(rounds['rounds'][round]))

                if rounds['rounds'][round] is None:
                    # print("Type None isn't valid.")
                    break

                if 'trump' not in rounds['rounds'][round]:
                    # print("Round hasn't a trump, so we skip it")
                    break

                for player in range(amount_players):
                    if 'hand' not in rounds['rounds'][round]['player'][player]:
                        # print("Round has no hands, so we skip it")
                        break
                    dealer_gift = rounds['rounds'][round]['player'][player]['hand']
                    player_cards = []
                    for c in dealer_gift:
                        player_cards.append(CardParser.create_card(c))
                    table.insert(player, player_cards)

                trumpf = rounds['rounds'][round]['trump']
                card_type = TrumpfColor()
                trumpf_message = Message(card_type, int(trumpf))
                game_type = trumpf_message.trumpf_parser()
                amount_stich = len(rounds['rounds'][round]['tricks'])

                if amount_stich == 0:
                    break
                for stich in range(amount_stich):
                    current_player = int(rounds['rounds'][round]['tricks'][stich]['first'])
                    for player_seat in range(amount_players):
                        played_card = rounds['rounds'][round]['tricks'][stich]['cards'][player_seat]
                        card = CardParser.create_card(played_card)
                        table[current_player].append(card)
                        current_player = (current_player - 1) % amount_players

                # Round complete with all hand cards for all players and trump
                print("trumpf: " + str(game_type.mode))
                print("player0: " + str(table[0]))
                print("player1: " + str(table[1]))
                print("player2: " + str(table[2]))
                print("player3: " + str(table[3]))

                for learning_player in range(amount_players):
                    # print("learning player: " + str(learning_player))

                    table_list = []
                    hand_list = []
                    trumpf_list = []
                    target_list = []

                    for stich in range(amount_stich):
                        cards_on_table = []
                        hand = table[learning_player][stich:amount_stich]
                        print("Stich: " + str(rounds['rounds'][round]['tricks'][stich]))
                        current_player = int(rounds['rounds'][round]['tricks'][stich]['first'])
                        # print("current Player: " + str(current_player))
                        player_seat = 0

                        while current_player != learning_player:
                            played_card = rounds['rounds'][round]['tricks'][stich]['cards'][player_seat]
                            card = CardParser.create_card(played_card)

                            cards_on_table.insert(player_seat, card)

                            current_player = (current_player - 1) % amount_players
                            player_seat += 1

                        target = rounds['rounds'][round]['tricks'][stich]['cards'][player_seat]
                        target_card = CardParser.create_card(target)

                        print("Cards on table: " + str(cards_on_table))
                        table_list.append(cards_on_table)
                        print("Handcards: " + str(hand))
                        hand_list.append(hand)
                        print("Trumpf: " + str(game_type.mode))
                        trumpf_list.append(game_type)
                        print("Target: " + str(target_card))
                        target_list.append(target_card)

                    # call BotNetwork with hand cards, cards from table, trumpf and the list of all targets
                    network.train_the_model(hand_list, table_list, trumpf_list, target_list)

        file_addition = str(file_number) + datetime.now().strftime("__%Y-%m-%d_%H%M%S")
        network.save_model("./config/game_network_model_" + file_addition + ".h5")
        network.save_model("./config/game_network_model_" + file_addition + ".json", True)
        network.save_weights("./config/game_network_weights_" + file_addition + ".h5")

        file_number += 1

    k.clear_session()


if __name__ == '__main__':
    start_training()
