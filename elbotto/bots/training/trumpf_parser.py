import glob
import json

from datetime import datetime
from keras import backend as k

from elbotto.bots.training import trumpf_training as traintrumpf
from elbotto.bots.training.card_parser import CardParser
from elbotto.bots.training.trumpf_converter import Message, TrumpfColor


def start_trumpf_training():
    # create an instance of the model to want to train
    network = traintrumpf.TrumpfTraining("Supervised_Trumpfnetwork")
    # Import and validate all dates
    files = glob.glob('./data/*.txt')
    for file_path in files:
        print(file_path)

        lines = open(file_path).readlines()

        hand_list = []
        trumpf_list = []

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

                trumpf = rounds['rounds'][round]['trump']
                card_type = TrumpfColor()
                trumpf_message = Message(card_type, int(trumpf))
                game_type = trumpf_message.trumpf_parser()

                # Round complete with all hand cards for all players and trump
                if game_type.mode == "TRUMPF":
                    print("trumpf: " + str(game_type.trumpf_color.name))
                else:
                    print("trumpf: " + str(game_type.mode))
                print("player0: " + str(table[0]))
                print("player1: " + str(table[1]))
                print("player2: " + str(table[2]))
                print("player3: " + str(table[3]))

                trumpf_decider = int(rounds['rounds'][round]['tricks'][0]['first'])
                print(str(trumpf_decider))
                hand_list.append(table[trumpf_decider])
                trumpf_list.append(game_type)

        # call BotNetwork with hand cards and the list of all targets trumpf
        network.train_the_model(hand_list, trumpf_list)

    file_addition = datetime.now().strftime("__%Y-%m-%d_%H%M%S")
    network.save_model("./config/trumpf_network_model" + file_addition + ".h5")
    network.save_model("./config/trumpf_network_model" + file_addition + ".json", True)
    network.save_weights("./config/trumpf_network_weights" + file_addition + ".h5")

    k.clear_session()


if __name__ == '__main__':
    start_trumpf_training()
