import glob
import json

from keras import backend as k

from elbotto.bots.training import training_trumpfnetwork as traintrumpf
from elbotto.bots.training.cardparser import CardParser
from elbotto.bots.training.trumpf_converter import Message, TrumpfColor


def start_trumpf_training():
    network = traintrumpf.Training("Supervised_Trumpfnetwork")
    files = glob.glob('./data/*.txt')
    for file_path in files:
        print(file_path)

        lines = open(file_path).readlines()

        hand_list = []
        trumpf_list = []

        for line in lines:
            tournament = line[43:]
            rounds = json.loads(tournament)

            print(rounds)
            print("Tournament: " + str(rounds))

            amount_rounds = len(rounds['rounds'])
            print("Anzahl Rounds: " + str(amount_rounds))
            amount_players = len(rounds['rounds'][0]['player'])

            for round in range(amount_rounds):
                table = []

                print(str(round) + ". Round: " + str(rounds['rounds'][round]))

                if rounds['rounds'][round] is None:
                    print("Type None isn't valid.")
                    break

                if 'trump' not in rounds['rounds'][round]:
                    print("Round hasn't a trump, so we skip it")
                    break

                for player in range(amount_players):
                    if 'hand' not in rounds['rounds'][round]['player'][player]:
                        print("Round has no hands, so we skip it")
                        break
                    dealer_gift = rounds['rounds'][round]['player'][player]['hand']
                    player_cards = []
                    for c in dealer_gift:
                        player_cards.append(CardParser.create_card(c))
                    table.insert(player, player_cards)

                print("Table before stich: " + str(table))

                amount_stich = len(rounds['rounds'][round]['tricks'])
                print(amount_stich)
                if amount_stich == 0:
                    break
                for stich in range(amount_stich):
                    current_player = int(rounds['rounds'][round]['tricks'][stich]['first'])
                    for player_seat in range(amount_players):
                        played_card = rounds['rounds'][round]['tricks'][stich]['cards'][player_seat]
                        card = CardParser.create_card(played_card)
                        table[current_player].append(card)
                        current_player = (current_player - 1) % amount_players

                trumpf = rounds['rounds'][round]['trump']
                card_type = TrumpfColor()
                trumpf_message = Message(card_type, int(trumpf))
                game_type = trumpf_message.trumpf_parser()

                # Round complete with all hand cards for all players and trump
                print("trumpf: " + str(game_type.mode))
                print("player0: " + str(table[0]))
                print("player1: " + str(table[1]))
                print("player2: " + str(table[2]))
                print("player3: " + str(table[3]))

                trumpf_decider = int(rounds['rounds'][round]['tricks'][0]['first'])
                hand_list.append(table[trumpf_decider])
                trumpf_list.append(game_type)

        network.train_the_model(hand_list, trumpf_list)

    network.save_model("./config/trumpf_network_model.h5")
    network.save_model("./config/trumpf_network_model.json", True)
    network.save_weights("./config/trumpf_network_weights.h5")

    k.clear_session()


if __name__ == '__main__':
    start_trumpf_training()
