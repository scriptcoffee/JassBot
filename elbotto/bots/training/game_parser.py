import json
from keras import backend as k
from elbotto.bots.training.card_parser import create_card
from elbotto.bots.training.game_training import GameTraining
from elbotto.bots.training.parser import get_trumpf, complete_hand_cards_with_stiches, get_remaining_hand_cards
from elbotto.bots.training.parser import check_path, check_file
from elbotto.bots.training.parser import print_trumpf, print_table


def start_game_training(data_path='./data/', data_file='*.txt', network_name='', log_path='./logs/game'):
    if check_path(data_path) is None:
        return
    files = check_file(data_path, data_file)
    if files is None:
        return
    if check_path(log_path) is None:
        return

    # create an instance of the model to want to train
    network = GameTraining(network_name, log_path)

    samples = extract_logfiles(files, network)

    k.clear_session()

    print_statistic(samples)


def print_statistic(samples):
    print("Total amount of stich: {}".format(samples))


def extract_logfiles(files, network):
    file_number = 0
    samples = 0
    for file_path in files:
        print(file_path)

        with open(file_path) as file_content:
            lines = file_content.readlines()

            samples = work_trought_each_line(lines, network, samples)

        network.save_model_and_weights("game", file_number)

        file_number += 1
    return samples


def work_trought_each_line(lines, network, samples):
    for line in lines:
        game = line[43:]
        rounds = json.loads(game)

        print("Game: {}".format(line))

        samples = collect_game_per_player(network, rounds, samples)

    return samples


def collect_game_per_player(network, rounds, samples):
    amount_rounds = len(rounds['rounds'])
    amount_players = len(rounds['rounds'][0]['player'])

    for learning_player in range(amount_players):
        table_list = []
        hand_list = []
        played_card_list = []
        trumpf_list = []
        target_list = []
        round_finish = False

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

            amount_stich = len(rounds['rounds'][round]['tricks'])
            for stich_number in range(amount_stich):
                hand = table[learning_player][stich_number:amount_stich]
                print("Stich: {}".format(rounds['rounds'][round]['tricks'][stich_number]))

                stich_record = rounds['rounds'][round]['tricks'][stich_number]
                cards_on_table, target_card = get_active_playing_cards(stich_record, amount_players, learning_player)

                played_cards = get_played_cards(amount_players, stich_number, table)

                played_card_list.append(played_cards)
                table_list.append(cards_on_table)
                hand_list.append(hand)
                trumpf_list.append(game_type)
                target_list.append(target_card)

            round_finish = True

        # call BotNetwork with hand cards, cards from table, trumpf and the list of all targets
        if round_finish:
            network.train_the_model(hand_list, table_list, played_card_list, trumpf_list, target_list)
            samples += len(hand_list)

    return samples


def get_active_playing_cards(stich, amount_players, learning_player):
    cards_on_table = []
    current_player = int(stich['first'])
    player_seat = 0

    while current_player != learning_player:
        played_card = stich['cards'][player_seat]
        card = create_card(played_card)

        cards_on_table.insert(player_seat, card)

        current_player = (current_player - 1) % amount_players
        player_seat += 1

    target_card = get_target_card(player_seat, stich)

    return cards_on_table, target_card


def get_target_card(player_seat, stich):
    target = stich['cards'][player_seat]
    target_card = create_card(target)
    return target_card


def get_played_cards(amount_players, stich_number, table):
    played_cards = []
    for player in range(amount_players):
        played_cards.insert(player, table[player][:stich_number])
    return played_cards


if __name__ == '__main__':
    start_game_training(network_name="Supervised_Gamenetwork")
