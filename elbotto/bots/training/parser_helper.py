import os
import glob
from elbotto.bots.training.card_parser import create_card
from elbotto.bots.training.trumpf_converter import trumpf_converter, TrumpfCard


def check_path(data_path):
    if not os.path.isdir(data_path):
        print("The given path doesn't exist.")
        return None
    return data_path


def check_file(data_path, data_file):
    files = glob.glob(os.path.join(data_path, data_file))
    file_list = []
    for file in files:
        if os.path.exists(file):
            file_list.append(file)
    if len(file_list) == 0:
        print("Your entry file doesn't exist.")
        file_list = None
    return file_list


def get_trumpf(round):
    if 'trump' not in round:
        # print("Round hasn't a trump, so we skip it")
        return None
    else:
        trumpf = round['trump']
        return trumpf_converter(trumpf)


def get_remaining_hand_cards(end_hand_list, amount_players, table):
    for player in range(amount_players):
        if 'hand' not in end_hand_list[player]:
            # print("Round has no hands, so we skip it")
            break
        dealer_gift = end_hand_list[player]['hand']
        player_cards = []
        for c in dealer_gift:
            player_cards.append(create_card(c))
        table.insert(player, player_cards)
    return table


def complete_hand_cards_with_stiches(stich_list, amount_players, table):
    amount_stich = len(stich_list)
    if amount_stich == 0 or len(table) == 0:
        # print("No stich exist or table is empty!")
        return 0
    for stich in range(amount_stich):
        current_player = int(stich_list[stich]['first'])
        for player_seat in range(amount_players):
            played_card = stich_list[stich]['cards'][player_seat]
            card = create_card(played_card)
            table[current_player].append(card)
            current_player = (current_player - 1) % amount_players
    return table


def print_trumpf(game_type):
    if not isinstance(game_type, TrumpfCard):
        # print("No trumpf exist!")
        return 0
    if game_type.mode == "TRUMPF":
        print("trumpf: {}".format(game_type.trumpf_color.name))
    else:
        print("trumpf: {}".format(game_type.mode))


def print_table(table):
    if isinstance(table, list):
        if len(table) == 0 or len(table[0]) == 0:
            # print("No players on the table or the players have no cards.")
            return 0
        else:
            print("player0: {}".format(table[0]))
            print("player1: {}".format(table[1]))
            print("player2: {}".format(table[2]))
            print("player3: {}".format(table[3]))
    else:
        # print("Table is not a list.")
        return 0


def print_training_time(start_time, end_time, network_name, save_in_file=True):
    output = "The training started at {} and ends at {}.".format(start_time, end_time)
    print(output)
    if save_in_file:
        filename = './logs/TimeMeasurement_{}_{}_{}{}{}.log'\
            .format(network_name[0:len(network_name)], start_time[0:10],
                    start_time[11:13], start_time[14:16], start_time[17:19])
        save_output(output, filename)


def save_output(output, filename='Unknown'):
    file = open(filename, 'w')
    file.write(output)
    file.close()
