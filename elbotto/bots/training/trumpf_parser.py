import glob
import json
from keras import backend as k
from elbotto.bots.training import trumpf_training as training_trumpf_network
from elbotto.bots.training.trumpf_converter import trumpf_converter
from elbotto.bots.training.parser import get_trumpf, complete_hand_cards_with_stiches, get_remaining_hand_cards
from elbotto.bots.training.parser import print_trumpf, print_table


def start_trumpf_training():
    # create an instance of the model to want to train
    network = training_trumpf_network.TrumpfTraining("Supervised_Trumpfnetwork")
    # Import and validate all dates
    files = glob.glob('./data/*.txt')
    for file_path in files:
        print(file_path)

        with open(file_path, 'r') as file_content:
            lines = file_content.readlines()

            hand_list = []
            trumpf_list = []

            for line in lines:
                game = line[43:]
                rounds = json.loads(game)

                print("Game: {}".format(line))

                amount_rounds = len(rounds['rounds'])
                amount_players = len(rounds['rounds'][0]['player'])

                for round in range(amount_rounds):

                    table = []

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

                    # Round complete with all hand cards for all players and trumpf
                    print_trumpf(game_type)
                    print_table(table)

                    trumpf_decider = int(rounds['rounds'][round]['tricks'][0]['first'])
                    if 'tss' in rounds['rounds'][round]:
                        hand_list.append(table[trumpf_decider])
                        trumpf_list.append(trumpf_converter(6))
                        trumpf_decider = (trumpf_decider + 2) % amount_players
                    hand_list.append(table[trumpf_decider])
                    trumpf_list.append(game_type)

            # call BotNetwork with hand cards and the list of all targets trumpf
            network.train_the_model(hand_list, trumpf_list)

    network.save_model_and_weights("trumpf")

    k.clear_session()


if __name__ == '__main__':
    start_trumpf_training()
