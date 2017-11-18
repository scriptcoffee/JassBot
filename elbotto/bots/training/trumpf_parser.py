import glob
import json
from datetime import datetime
from keras import backend as k
from elbotto.bots.training import trumpf_training as training_trumpf_network
from elbotto.bots.training.parser import get_trumpf, complete_hand_cards_with_stiches, get_remaining_hand_cards
from elbotto.bots.training.parser import print_trumpf, print_table


def start_trumpf_training():
    # create an instance of the model to want to train
    network = training_trumpf_network.TrumpfTraining("Supervised_Trumpfnetwork")
    # Import and validate all dates
    files = glob.glob('./data/MLAI_8-1_log.txt')
    for file_path in files:
        print(file_path)

        with open(file_path, 'r') as file_content:
            lines = file_content.readlines()

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

                    game_type = get_trumpf(rounds['rounds'][round])
                    if game_type is None:
                        break

                    table = get_remaining_hand_cards(rounds['rounds'][round]['player'], amount_players, table)
                    table = complete_hand_cards_with_stiches(rounds['rounds'][round]['tricks'], amount_players, table)
                    if table == 0:
                        break

                    # Round complete with all hand cards for all players and trump
                    print_trumpf(game_type)
                    print_table(table)

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
