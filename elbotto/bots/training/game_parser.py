import os
import json
from datetime import datetime
from keras import backend as k
from elbotto.bots.training.game_training import GameTraining
from elbotto.bots.training.card_parser import create_card
from elbotto.bots.training.parser import get_trumpf, complete_hand_cards_with_stiches, get_remaining_hand_cards
from elbotto.bots.training.parser import check_path, check_file
from elbotto.bots.training.parser import print_trumpf, print_table


def start_game_training(data_path='./data/', data_file='MLAI_8-1_log.txt', network_name='', log_path='./logs'):
    os.chdir(os.path.dirname(__file__))
    if check_path(data_path) is None:
        return
    files = check_file(data_path, data_file)
    if files is None:
        return
    if check_path(log_path) is None:
        return

    # create an instance of the model to want to train
    network = GameTraining(network_name, log_path)

    file_number = 0
    samples = 0
    for file_path in files:
        print(file_path)

        with open(file_path) as file_content:
            lines = file_content.readlines()

            for line in lines:
                game = line[43:]
                rounds = json.loads(game)

                print("Game: {}".format(line))

                amount_rounds = len(rounds['rounds'])
                amount_players = len(rounds['rounds'][0]['player'])

                for learning_player in range(amount_players):

                    table_list = []
                    hand_list = []
                    trumpf_list = []
                    target_list = []
                    round_finish = False

                    for round in range(amount_rounds):

                        table = []

                        print("{}. Round: {}".format(round, rounds['rounds'][round]))

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

                        # print("learning player: {}".format(learning_player))

                        amount_stich = len(rounds['rounds'][round]['tricks'])
                        for stich in range(amount_stich):
                            cards_on_table = []
                            hand = table[learning_player][stich:amount_stich]
                            print("Stich: {}".format(str(rounds['rounds'][round]['tricks'][stich])))
                            current_player = int(rounds['rounds'][round]['tricks'][stich]['first'])
                            # print("current Player: {}".format(current_player))
                            player_seat = 0

                            while current_player != learning_player:
                                played_card = rounds['rounds'][round]['tricks'][stich]['cards'][player_seat]
                                card = create_card(played_card)

                                cards_on_table.insert(player_seat, card)

                                current_player = (current_player - 1) % amount_players
                                player_seat += 1

                            target = rounds['rounds'][round]['tricks'][stich]['cards'][player_seat]
                            target_card = create_card(target)

                            print("Cards on table: {}".format(cards_on_table))
                            table_list.append(cards_on_table)
                            print("Handcards: {}".format(hand))
                            hand_list.append(hand)
                            print("Trumpf: {}".format(game_type.mode))
                            trumpf_list.append(game_type)
                            print("Target: {}".format(target_card))
                            target_list.append(target_card)

                        round_finish = True

                    # call BotNetwork with hand cards, cards from table, trumpf and the list of all targets
                    if round_finish:
                        network.train_the_model(hand_list, table_list, trumpf_list, target_list)
                        samples += len(hand_list)

        file_addition = "{}{}".format(file_number, datetime.now().strftime("__%Y-%m-%d_%H%M%S"))
        network.save_model("./config/game_network_model_{}.h5".format(file_addition))
        network.save_model("./config/game_network_model_{}.json".format(file_addition), True)
        network.save_weights("./config/game_network_weights_{}.h5".format(file_addition))

        file_number += 1

    k.clear_session()

    print(samples)


if __name__ == '__main__':
    start_game_training(network_name="Supervised_Gamenetwork")
