import json
import time
from keras import backend as k
from elbotto.bots.training.trumpf_training import TrumpfTraining
from elbotto.bots.training.trumpf_converter import trumpf_converter
from elbotto.bots.training.parser_helper import get_trumpf, complete_hand_cards_with_stiches, get_remaining_hand_cards
from elbotto.bots.training.parser_helper import check_path, check_file
from elbotto.bots.training.parser_helper import print_trumpf, print_table, print_training_time


def start_trumpf_training(data_path='./data/BA_Logfiles/', data_file='*.txt', network_name='', log_path='./logs/trumpf'):
    if check_path(data_path) is None:
        return
    files = check_file(data_path, data_file)
    if files is None:
        return
    if check_path(log_path) is None:
        return

    # create an instance of the model to want to train
    start_time = time.strftime("%d.%m.%Y %H:%M:%S")
    network = TrumpfTraining(network_name, log_path)

    trumpf_tuples, tss = extract_logfiles(files, network)

    network.save_model_and_weights("trumpf", "final")

    k.clear_session()

    print_statistics(trumpf_tuples, tss)
    end_time = time.strftime("%d.%m.%Y %H:%M:%S")
    print_training_time(start_time, end_time, network_name)


def print_statistics(trumpf_tuples, tss):
    for t in trumpf_tuples:
        print("{}: {}".format(t, trumpf_tuples[t]))
    print("SCHIEBEN: {}".format(tss))


def extract_logfiles(files, network):
    trumpf_tuples = {}
    tss = 0
    for file_path in files:
        print(file_path)

        with open(file_path, 'r') as file_content:
            lines = file_content.readlines()

            hand_list, trumpf_list, trumpf_tuples, tss = work_trought_each_line(lines, trumpf_tuples, tss)

            # call BotNetwork with hand cards and the list of all targets trumpf
            network.train_the_model(hand_list, trumpf_list)
    return trumpf_tuples, tss


def work_trought_each_line(lines, trumpf_tuples, tss):
    hand_list = []
    trumpf_list = []
    for line in lines:
        game = line[43:]
        rounds = json.loads(game)

        print("Game: {}".format(line))

        hand_list, trumpf_list, trumpf_tuples, tss = collect_trumpf_per_game(hand_list, rounds, trumpf_list,
                                                                             trumpf_tuples, tss)
    return hand_list, trumpf_list, trumpf_tuples, tss


def collect_trumpf_per_game(hand_list, rounds, trumpf_list, trumpf_tuples, tss):
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

        trumpf_tuples = count_game_type(game_type, trumpf_tuples)

        trumpf_decider = int(rounds['rounds'][round]['tricks'][0]['first'])
        if 'tss' in rounds['rounds'][round]:
            if tss % 3 != 0:
                hand_list.append(table[trumpf_decider])
                trumpf_list.append(trumpf_converter(6))
            trumpf_decider = (trumpf_decider + 2) % amount_players
            tss += 1
        hand_list.append(table[trumpf_decider])
        trumpf_list.append(game_type)
    return hand_list, trumpf_list, trumpf_tuples, tss


def count_game_type(game_type, trumpf_tuples):
    if game_type.mode == "TRUMPF":
        if game_type.trumpf_color.name not in trumpf_tuples:
            trumpf_tuples[game_type.trumpf_color.name] = 0
        counter = int(trumpf_tuples[game_type.trumpf_color.name])
        trumpf_tuples[game_type.trumpf_color.name] = (counter + 1)
    else:
        if game_type.mode not in trumpf_tuples:
            trumpf_tuples[game_type.mode] = 0
        counter = int(trumpf_tuples[game_type.mode])
        trumpf_tuples[game_type.mode] = (counter + 1)
    return trumpf_tuples


if __name__ == '__main__':
    start_trumpf_training(network_name="Supervised Trumpfnetwork")
